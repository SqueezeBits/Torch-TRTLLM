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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

from loguru import logger
from torch.fx import GraphModule
from torch.fx.node import Node

from ....types import StrictlyTyped
from .graph_pass import GraphOptimizationPass
from .pass_result import PassResult


class NodewisePassResult(StrictlyTyped, ABC):
    require_fake_tensor_prop: bool = False

    @abstractmethod
    def apply(self, node: Node, pass_name: str) -> bool:
        ...


class ModifiedInsideThePass(NodewisePassResult):
    def apply(self, node: Node, pass_name: str) -> bool:
        if node.stack_trace:
            node.stack_trace = f"{node.stack_trace} -> modified inside {pass_name}"
        return True


def always(_: Node) -> Literal[True]:
    return True


class ReplaceAllUses(NodewisePassResult):
    by: Node
    replace_user_only_if: Callable[[Node], bool] = always
    propagate_meta: bool = False

    def apply(self, node: Node, pass_name: str) -> bool:
        if not self.propagate_meta and self.by.stack_trace:
            self.by.stack_trace = (
                f"{self.by.stack_trace} -> replaced all uses of {node.format_node()} after {pass_name}"
            )
        replaced_nodes = node.replace_all_uses_with(
            self.by,
            self.replace_user_only_if,
            propagate_meta=self.propagate_meta,
        )
        return len(replaced_nodes) > 0


class ReplaceAmongInputs(NodewisePassResult):
    occurrences_of: Node
    by: Node

    def apply(self, node: Node, pass_name: str) -> bool:
        if self.by.stack_trace:
            self.by.stack_trace = (
                f"{self.by.stack_trace} "
                f"-> replaced all occurrences of {self.occurrences_of} in {node.format_node()} after {pass_name}"
            )
        node.replace_input_with(self.occurrences_of, self.by)
        return True


class NodewiseOptimizationPass(GraphOptimizationPass):
    """Abstract class for implementing node-wise rewriting pass."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Apply `cls.rewrite` method across all nodes in the graph.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        modified = False
        require_fake_tensor_prop = False
        nodes = list(graph_module.graph.nodes)
        for node in nodes:
            for src, result in self.rewrite(node).items():
                is_applied = result.apply(src, pass_name=type(self).__name__)
                logger.trace(f"[{type(self).__name__}] {src}: {type(result).__name__}({result}) ({is_applied=})")
                modified = modified or is_applied
                require_fake_tensor_prop = require_fake_tensor_prop or result.require_fake_tensor_prop

        return PassResult(
            graph_module=graph_module,
            modified=modified,
            require_fake_tensor_prop=require_fake_tensor_prop,
        )

    @abstractmethod
    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Rewrite the given node.

        Args:
            node (Node): a node to rewrite

        Returns:
            dict[Node, NodewisePassResult]: a dictionary mapping an existing node to its nodewise pass result.
        """
