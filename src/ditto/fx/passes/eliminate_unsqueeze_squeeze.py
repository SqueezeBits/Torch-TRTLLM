from torch.fx import Node

from ..nodes import SqueezeDim, Unsqueeze
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateUnsqueezeSqueeze(NodeWiseOptimizationPass):
    """Eliminate unsqueeze followed by a squeeze with the same dim."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (squeeze := SqueezeDim.specialize_from(node))
            and (unsqueeze := Unsqueeze.specialize_from(squeeze.this))
            and (squeeze_dim := squeeze.nonnegative_dim) is not None
            and (unsqueeze_dim := unsqueeze.nonnegative_dim) is not None
            and squeeze_dim == unsqueeze_dim
        ):
            return {}
        return {node: unsqueeze.this}
