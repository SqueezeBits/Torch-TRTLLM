from abc import abstractmethod

from torch.fx import GraphModule
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassResult

from .graph_pass import GraphOptimizationPass


class NodeWiseOptimizationPass(GraphOptimizationPass):
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
            if replacement_map := self.rewrite(node):
                is_replaced = [
                    (
                        original_node is replacement_node
                        or len(original_node.replace_all_uses_with(replacement_node)) > 0
                    )
                    for original_node, replacement_node in replacement_map.items()
                ]
                modified = modified or any(is_replaced)

        return PassResult(graph_module, modified)

    @abstractmethod
    def rewrite(self, node: Node) -> dict[Node, Node]:
        """Rewrite the given node.

        Args:
            node (Node): a node to rewrite

        Returns:
            dict[Node, Node]: a dictionary mapping an existing node to its replacement.
        """
