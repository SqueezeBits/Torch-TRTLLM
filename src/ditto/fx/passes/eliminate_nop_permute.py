from torch.fx import Node

from ..nodes import Permute
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateNopPermute(NodewiseOptimizationPass):
    """Eliminate permute whose axis permutation is trivial."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((permute := Permute.specialize_from(node)) and permute.dims == [*range(permute.ndim)]):
            return {}
        return {node: ReplaceAllUses(by=permute.this)}
