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

from typing import Any

import torch
import torch.utils._pytree as pytree
from loguru import logger
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from ...utils import get_fake_mode


def cleanup(graph_module: GraphModule, run_fake_tensor_prop: bool = False) -> None:
    """Clean up the graph module.

    Args:
        graph_module (GraphModule): The graph module to clean up.
        run_fake_tensor_prop (bool): Whether to run FakeTensorProp.
    """
    graph = graph_module.graph
    graph.eliminate_dead_code()
    graph.lint()
    if run_fake_tensor_prop:
        fake_tensor_prop(graph_module)
    graph_module.recompile()


def fake_tensor_prop(graph_module: GraphModule) -> None:
    """Run FakeTensorProp on the graph module.

    Args:
        graph_module (GraphModule): The graph module to run FakeTensorProp on.
    """
    if fake_mode := get_fake_mode(graph := graph_module.graph):
        logger.debug("Running FakeTensorProp")
        interpreter = FakeTensorPropOnCPU(graph_module, fake_mode)
        _ = interpreter.propagate(*(placeholder.meta["val"] for placeholder in graph.find_nodes(op="placeholder")))


class FakeTensorPropOnCPU(FakeTensorProp):
    """Run FakeTensorProp on the graph module on CPU."""

    def fetch_args_kwargs_from_env(self, n: Node) -> tuple[tuple[Any], dict[str, Any]]:
        args, kwargs = super().fetch_args_kwargs_from_env(n)
        return pytree.tree_map(move_to_cpu_if_tensor, (args, kwargs))

    def run_node(self, n: Node) -> Any:
        if (val := n.meta.get("val")) is not None:
            return val
        val = super().run_node(n)
        logger.trace(f"FakeTensorProp: {n.format_node()} -> {val}")
        return val


def move_to_cpu_if_tensor(leaf: Any) -> Any:
    """Move the leaf to the CPU if it is a tensor.

    Args:
        leaf (Any): The leaf to move to the CPU if it is a tensor.

    Returns:
        Any: The given leaf to be moved to the CPU or not.
    """
    if isinstance(leaf, torch.Tensor) and leaf.device != torch.device("cpu"):
        return leaf.cpu()
    return leaf
