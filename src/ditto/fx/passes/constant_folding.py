# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
from itertools import chain

import torch
import torch.utils._pytree as pytree
from loguru import logger
from torch.fx import GraphModule, Node

from ..nodes import ATenOp, GetAttr
from .infra import GraphOptimizationPass, PassResult, propagate_metadata_from
from .infra.cleanup import cleanup


# TODO: fix memory leak from constant folding
class ConstantFolding(GraphOptimizationPass):
    """Fold constant nodes in the graph."""

    @torch.inference_mode()  # type: ignore[misc]
    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        foldable_nodes: dict[Node, torch.Tensor] = {}

        def get_qualname() -> str:
            i = 0
            qualname = "folded_constant"
            while hasattr(graph_module, qualname):
                i += 1
                qualname = f"folded_constant_{i}"
            return qualname

        def are_all_input_nodes_fetchable(n: Node) -> bool:
            nonlocal foldable_nodes
            return all(
                (input_node in foldable_nodes or GetAttr.specialize_from(input_node) is not None)
                for input_node in n.all_input_nodes
            )

        def fetch_value(n: Node) -> torch.Tensor:
            nonlocal foldable_nodes
            return get_attr.tensor if (get_attr := GetAttr.specialize_from(n)) else foldable_nodes[n]

        def is_foldable(node: Node) -> bool:
            nonlocal foldable_nodes
            if node in foldable_nodes:
                return True

            if (aten_op := ATenOp.specialize_from(node)) and are_all_input_nodes_fetchable(node):
                flat_inputs, spec = pytree.tree_flatten((node.args, node.kwargs))
                flat_values = tuple(fetch_value(arg) if isinstance(arg, Node) else arg for arg in flat_inputs)
                arg_values, kwarg_values = pytree.tree_unflatten(flat_values, spec)
                foldable_nodes[node] = aten_op.target(*arg_values, **kwarg_values)
                return True

            return False

        nodes_to_replace = [
            node for node in graph.nodes if is_foldable(node) and all(not is_foldable(user) for user in node.users)
        ]
        if not nodes_to_replace:
            return PassResult(graph_module=graph_module, modified=False)

        for node in nodes_to_replace:
            name = get_qualname()
            with graph.inserting_after(node):
                get_attr = GetAttr.create(graph, name, foldable_nodes.pop(node))
                propagate_metadata_from(node, to=get_attr)
                node.replace_all_uses_with(get_attr.node)
                graph.erase_node(node)

        if nodes_to_replace:
            cleanup(graph_module)

        del foldable_nodes, nodes_to_replace
        logger.debug(f"{gc.collect()=}")

        return PassResult(graph_module=remove_unused_constants(graph_module), modified=True)


def remove_unused_constants(graph_module: GraphModule) -> GraphModule:
    """Remove unused constants from the graph module.

    Note that this function modifies graph module in-place.

    Args:
        graph_module (GraphModule): The graph module to process

    Returns:
        GraphModule: The graph module given as input
    """
    referenced_attr_names = {
        target for node in graph_module.graph.nodes if node.op == "get_attr" and isinstance(target := node.target, str)
    }
    unused_attr_names = [
        name for name in chain(graph_module._parameters, graph_module._buffers) if name not in referenced_attr_names
    ]
    for name in unused_attr_names:
        delattr(graph_module, name)

    logger.debug(f"Removed {len(unused_attr_names)} unused attributes {gc.collect()=}")
    return graph_module
