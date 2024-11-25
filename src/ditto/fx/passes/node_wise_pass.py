from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

from loguru import logger
from torch.fx import GraphModule
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassResult

from ...types import StrictlyTyped
from .graph_pass import GraphOptimizationPass


class NodewisePassResult(StrictlyTyped, ABC):
    @abstractmethod
    def apply(self, node: Node) -> bool:
        ...


class ModifiedInsideThePass(NodewisePassResult):
    def apply(self, node: Node) -> bool:
        return True


def always(_: Node) -> Literal[True]:
    return True


class ReplaceAllUses(NodewisePassResult):
    by: Node
    replace_user_only_if: Callable[[Node], bool] = always
    propagate_meta: bool = False

    def apply(self, node: Node) -> bool:
        replaced_nodes = node.replace_all_uses_with(
            self.by,
            self.replace_user_only_if,
            propagate_meta=self.propagate_meta,
        )
        return len(replaced_nodes) > 0


class ReplaceAmongInputs(NodewisePassResult):
    occurences_of: Node
    by: Node

    def apply(self, node: Node) -> bool:
        node.replace_input_with(self.occurences_of, self.by)
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
        nodes = list(graph_module.graph.nodes)
        for node in nodes:
            for src, result in self.rewrite(node).items():
                is_applied = result.apply(src)
                logger.debug(f"{src}: {type(result).__name__}({result}) (success: {is_applied})")
                modified = modified or is_applied

        return PassResult(graph_module, modified)

    @abstractmethod
    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Rewrite the given node.

        Args:
            node (Node): a node to rewrite

        Returns:
            dict[Node, NodewisePassResult]: a dictionary mapping an existing node to its nodewise pass result.
        """
