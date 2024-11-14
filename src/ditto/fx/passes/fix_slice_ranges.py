from torch.fx import Node

from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import SliceNode


class FixSliceRanges(NodeWiseOptimizationPass):
    """Fix the slice end value if it is the int64 max value."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (s := SliceNode.specialize_from(node)) and s.end == ((1 << 63) - 1) and (dim_size := s.dim_size) is not None
        ):
            return {}
        node.args = node.args[:3] + (dim_size,) + node.args[4:]
        return {node: node}
