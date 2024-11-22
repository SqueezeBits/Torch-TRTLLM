from torch.fx import Node

from ..nodes import Slice
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateNopSlice(NodeWiseOptimizationPass):
    """Eliminate slice that has no effect."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not ((s := Slice.specialize_from(node)) and s.start == 0 and s.end == ((1 << 63) - 1) and s.step == 1):
            return {}
        return {node: s.this}
