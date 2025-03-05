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

from collections.abc import Generator

import torch
from torch.fx import GraphModule, Node

from ..nodes import Cat, GetAttr
from ..targets import Dequantize
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class FuseDequantizes(NodewiseOptimizationPass):
    """Fuse consecutive dequantize nodes with cat node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (cat := Cat.specialize_from(node)) and len(dequantizes := cat.tensors) > 1 and are_fusible(dequantizes)
        ):
            return {}

        name_gen = generate_name(node.graph.owning_module)
        input_nodes: list[Node] = []
        output_shapes: list[torch.Size] = []
        for zipped_input_nodes in zip(*[dequantize.all_input_nodes for dequantize in dequantizes]):
            attrs = [attr for node in zipped_input_nodes if (attr := GetAttr.specialize_from(node)) is not None]
            fused_tensor = torch.cat([attr.tensor for attr in attrs], dim=1)
            with node.graph.inserting_before(min(zipped_input_nodes)):
                fused_tensor_attr = GetAttr.create(node.graph, next(name_gen), fused_tensor)
            propagate_metadata_from(*attrs, to=fused_tensor_attr)
            input_nodes.append(fused_tensor_attr.node)
            output_shapes.append(fused_tensor_attr.tensor.shape)

        assert isinstance(org_dequantize := dequantizes[0].target, Dequantize)
        dequantize = org_dequantize.model_copy(update={"output_shape": output_shapes[0]})
        with node.graph.inserting_before(node):
            dequantize_node = node.graph.call_function(dequantize, args=tuple(input_nodes))
        nodes_to_replace = [node] + list(dequantizes)
        propagate_metadata_from(*nodes_to_replace, to=dequantize_node)

        return {node: ReplaceAllUses(by=dequantize_node)}


def are_fusible(dequantizes: list[Node]) -> bool:
    """Check if the dequantize nodes are fusible.

    Args:
        dequantizes (list[Node]): A list of dequantize nodes to check for fusibility

    Returns:
        bool: True if all dequantize nodes are fusible, False otherwise
    """
    if not (
        all(isinstance(dequantize.target, Dequantize) for dequantize in dequantizes)
        and all(
            len(dequantizes[0].all_input_nodes) == len(dequantize.all_input_nodes) for dequantize in dequantizes[1:]
        )
    ):
        return False

    for zipped_input_nodes in zip(*[dequantize.all_input_nodes for dequantize in dequantizes]):
        if not (
            (attrs := [attr for node in zipped_input_nodes if (attr := GetAttr.specialize_from(node)) is not None])
            and len(attrs) == len(zipped_input_nodes)
            and all(
                attrs[0].tensor.shape[0] == attr.tensor.shape[0] and attrs[0].tensor.dtype == attr.tensor.dtype
                for attr in attrs[1:]
            )
        ):
            return False

    return True


def generate_name(graph_module: GraphModule) -> Generator[str, None, None]:
    """Generate a unique name for the fused dequantize node.

    Args:
        graph_module (GraphModule): The graph module to generate the name for

    Returns:
        Generator[str, None, None]: A generator of unique names
    """
    name = "dequantize_fused_constant"
    if not hasattr(graph_module, name):
        yield name

    idx = 1
    while True:
        if not hasattr(graph_module, f"{name}_{idx}"):
            yield f"{name}_{idx}"
        idx += 1
