from torch.fx import Node

from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import SqueezeDimNode, UnsqueezeNode


class EliminateUnsqueezeSqueeze(NodeWiseOptimizationPass):
    """Eliminate unsqueeze followed by a squeeze with the same dim."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (squeeze := SqueezeDimNode.specialize_from(node))
            and (unsqueeze := UnsqueezeNode.specialize_from(squeeze.x))
            and (squeeze_dim := squeeze.nonnegative_dim) is not None
            and (unsqueeze_dim := unsqueeze.nonnegative_dim) is not None
            and squeeze_dim == unsqueeze_dim
        ):
            return {}
        return {node: unsqueeze.x}
