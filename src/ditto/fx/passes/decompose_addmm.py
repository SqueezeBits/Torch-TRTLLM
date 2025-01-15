from torch.fx import Node

from ..nodes import MM, AddMM, AddTensor
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class DecomposeAddMM(NodewiseOptimizationPass):
    """Decompose addmm into mm and add."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (addmm := AddMM.specialize_from(node)):
            return {}

        with (graph := node.graph).inserting_before(node):
            mm = MM.create(graph, addmm.mat1, addmm.mat2)
            inject_stack_trace_from(addmm, to=mm)
            add = AddTensor.create(graph, mm.node, addmm.this)
            inject_stack_trace_from(addmm, to=add)
        return {node: ReplaceAllUses(by=add.node)}
