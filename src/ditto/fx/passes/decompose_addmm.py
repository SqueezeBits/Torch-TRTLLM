from torch.fx import Node

from ..nodes import MM, Add, AddMM
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class DecomposeAddMM(NodewiseOptimizationPass):
    """Decompose addmm into mm and add."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (addmm := AddMM.specialize_from(node)):
            return {}

        with (graph := node.graph).inserting_before(node):
            mm = MM.create(graph, addmm.mat1, addmm.mat2)
            add = Add.create(graph, mm.node, addmm.this)

        return {node: ReplaceAllUses(by=add.node)}
