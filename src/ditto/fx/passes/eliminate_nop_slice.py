from torch.fx import Node

from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import SliceNode


class EliminateNopSlice(NodeWiseOptimizationPass):
    """Eliminate slice that has no effect."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (slice := SliceNode.specialize_from(node))
            and slice.start == 0
            and slice.end == ((1 << 63) - 1)
            and slice.step == 1
        ):
            return {}
        return {node: slice.x}
