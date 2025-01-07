from torch.fx import Node

from ..nodes import Cat, Slice
from ..nodes.aten.utils import has_same_values
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseConsecutiveSliceConcat(NodewiseOptimizationPass):
    """Fuse consecutive slices and concat that is identical to nop."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (cat_node := Cat.specialize_from(node))
            and (slice_nodes := [s for x in cat_node.tensors if (s := Slice.specialize_from(x))])
            and len(slice_nodes) == len(cat_node.tensors)
            and has_same_values(slice_nodes[0].nonnegative_dim, cat_node.nonnegative_dim)
            and Slice.are_consecutive(slice_nodes)
        ):
            return {}
        return {cat_node.node: ReplaceAllUses(by=slice_nodes[0].this)}
