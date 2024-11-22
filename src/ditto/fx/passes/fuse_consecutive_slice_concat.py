from torch.fx import Node

from ..nodes import Cat, Slice
from .node_wise_pass import NodeWiseOptimizationPass


class FuseConsecutiveSliceConcat(NodeWiseOptimizationPass):
    """Fuse consecutive slices and concat that is identical to nop."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (cat_node := Cat.specialize_from(node))
            and (slice_nodes := [s for x in cat_node.tensors if (s := Slice.specialize_from(x))])
            and len(slice_nodes) == len(cat_node.tensors)
            and has_same_values(slice_nodes[0].nonnegative_dim, cat_node.nonnegative_dim)
            and are_consecutive(slice_nodes)
        ):
            return {}
        return {cat_node.node: slice_nodes[0].this}


def are_consecutive(slice_nodes: list[Slice]) -> bool:
    return (
        len({s.this for s in slice_nodes}) == 1
        and len(dim_sizes := {s.dim_size for s in slice_nodes}) == 1
        and None not in dim_sizes
        and all(s.step == 1 for s in slice_nodes)
        and all(
            has_same_values(slice_nodes[i].nonnegative_end, slice_nodes[i + 1].nonnegative_start)
            for i in range(len(slice_nodes) - 1)
        )
        and has_same_values(slice_nodes[-1].nonnegative_end, slice_nodes[0].dim_size)
    )


def has_same_values(x: int | None, y: int | None) -> bool:
    if x is None or y is None:
        return False
    return x == y
