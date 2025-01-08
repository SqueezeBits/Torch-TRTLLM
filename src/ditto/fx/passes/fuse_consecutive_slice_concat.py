from torch.fx import Node

from ..nodes import Cat, Slice
from ..nodes.aten.utils import has_same_values
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseConsecutiveSliceConcat(NodewiseOptimizationPass):
    """Fuse consecutive slices and concat that is identical to nop."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (cat := Cat.specialize_from(node))
            and (slices := Slice.sort([s for x in cat.tensors if (s := Slice.specialize_from(x))]))
            and len(slices) == len(cat.tensors)
            and has_same_values(slices[0].nonnegative_dim, cat.nonnegative_dim)
            and Slice.are_consecutive(slices)
        ):
            return {}
        return {cat.node: ReplaceAllUses(by=slices[0].this)}
