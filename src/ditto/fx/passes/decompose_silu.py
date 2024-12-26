from torch.fx import Node

from ..nodes import Mul, Sigmoid, Silu
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class DecomposeSiLU(NodewiseOptimizationPass):
    """Decompose silu into sigmoid and mul."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((silu := Silu.specialize_from(node)) and (graph := node.graph)):
            return {}

        with graph.inserting_before(node):
            sigmoid = Sigmoid.create(graph, silu.this)
            mul = Mul.create(graph, silu.this, sigmoid.node)

        if nn_module_stack := node.meta["nn_module_stack"]:
            sigmoid.node.meta["nn_module_stack"] = nn_module_stack
            mul.node.meta["nn_module_stack"] = nn_module_stack

        return {node: ReplaceAllUses(by=mul.node)}
