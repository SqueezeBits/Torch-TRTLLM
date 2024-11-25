from torch.fx import Node

from ..nodes import Permute
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateNopPermute(NodeWiseOptimizationPass):
    """Eliminate permute whose axis permutation is trivial."""

    def rewrite(self, node: Node) -> dict[Node, Node]:
        if not ((permute := Permute.specialize_from(node)) and permute.dims == [*range(permute.ndim)]):
            return {}
        return {node: permute.this}
