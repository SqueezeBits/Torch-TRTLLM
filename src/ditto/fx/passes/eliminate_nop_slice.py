from torch.fx import Node

from ..nodes import Slice
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateNopSlice(NodewiseOptimizationPass):
    """Eliminate slice that has no effect."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((s := Slice.specialize_from(node)) and s.start == 0 and s.end == ((1 << 63) - 1) and s.step == 1):
            return {}
        return {node: ReplaceAllUses(by=s.this)}
