from torch.fx import Node

from ..nodes import PermuteNode
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateNopPermute(NodeWiseOptimizationPass):
    """Eliminate permute whose axis permutation is trivial."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not ((permute := PermuteNode.specialize_from(node)) and permute.dims == [*range(permute.ndim)]):
            return {}
        return {node: permute.x}
