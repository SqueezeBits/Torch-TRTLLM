from torch.fx import Node

from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import PermuteNode


class EliminateNopPermute(NodeWiseOptimizationPass):
    """Eliminate permute whose axis permutation is trivial."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not ((permute := PermuteNode.specialize_from(node)) and permute.dims == [*range(permute.ndim)]):
            return {}
        return {node: permute.x}
