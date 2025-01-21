import torch
from torch.fx import Node

from ..nodes import PowTensorScalar
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class RewritePowAsMulSelf(NodewiseOptimizationPass):
    """Rewrite pow op as mul op with self.

    Required to prevent engine build failures due to casts inserted where computations are in bf but literals remain fp.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (power := PowTensorScalar.specialize_from(node)) and power.other == 2:
            graph = node.graph

            with graph.inserting_before(node):
                equivalent_mul = graph.call_function(torch.ops.aten.mul.Tensor, (power.this, power.this))
                inject_stack_trace_from(node, to=equivalent_mul)

            return {node: ReplaceAllUses(by=equivalent_mul)}
        return {}
