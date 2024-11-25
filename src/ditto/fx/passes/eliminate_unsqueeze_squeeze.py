from torch.fx import Node

from ..nodes import SqueezeDim, Unsqueeze
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateUnsqueezeSqueeze(NodewiseOptimizationPass):
    """Eliminate unsqueeze followed by a squeeze with the same dim."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (squeeze := SqueezeDim.specialize_from(node))
            and (unsqueeze := Unsqueeze.specialize_from(squeeze.this))
            and (squeeze_dim := squeeze.nonnegative_dim) is not None
            and (unsqueeze_dim := unsqueeze.nonnegative_dim) is not None
            and squeeze_dim == unsqueeze_dim
        ):
            return {}
        return {node: ReplaceAllUses(by=unsqueeze.this)}
