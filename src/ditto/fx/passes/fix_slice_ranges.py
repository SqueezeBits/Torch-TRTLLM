from torch.fx import Node

from ..nodes import Slice
from .node_wise_pass import NodeWiseOptimizationPass


class FixSliceRanges(NodeWiseOptimizationPass):
    """Fix the slice end value if it is the int64 max value."""

    def rewrite(self, node: Node) -> dict[Node, Node]:
        if not (
            (s := Slice.specialize_from(node)) and s.end == ((1 << 63) - 1) and (dim_size := s.dim_size) is not None
        ):
            return {}
        node.args = node.args[:3] + (dim_size,) + node.args[4:]
        return {node: node}
