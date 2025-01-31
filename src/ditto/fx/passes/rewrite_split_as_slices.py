from torch.fx.node import Node

from ..nodes import GetItem, Slice, SplitTensor
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class RewriteSplitAsSlices(NodewiseOptimizationPass):
    """Rewrite a split node as a group of slices."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (split := SplitTensor.specialize_from(node))
            and isinstance(size := split.split_size, int)
            and (getitems := [getitem for user in node.users if (getitem := GetItem.specialize_from(user))])
            and len(getitems) == len(node.users)
        ):
            return {}

        graph = node.graph
        results: dict[Node, NodewisePassResult] = {}
        with graph.inserting_before(node):
            for getitem in getitems:
                s = Slice.create(
                    graph,
                    split.this,
                    split.dim,
                    getitem.idx * size,
                    (getitem.idx + 1) * size,
                )
                propagate_metadata_from(node, to=s)
                results[getitem.node] = ReplaceAllUses(by=s.node)
        return results
