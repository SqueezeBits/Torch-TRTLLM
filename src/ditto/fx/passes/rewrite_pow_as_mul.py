from torch.fx import Node

from ..nodes import MulTensor, PowTensorScalar
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class RewritePowAsMul(NodewiseOptimizationPass):
    """Rewrite pow op as mul op with self.

    Required to prevent engine build failures due to casts inserted where computations are in bf but literals remain fp.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (power := PowTensorScalar.specialize_from(node)) and power.other == 2:
            graph = node.graph

            with graph.inserting_before(node):
                equivalent_mul = MulTensor.create(graph, power.this, power.this)
                inject_stack_trace_from(node, to=equivalent_mul.node)

            return {node: ReplaceAllUses(by=equivalent_mul.node)}
        return {}
