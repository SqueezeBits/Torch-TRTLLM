import torch
from torch.fx import Node

from ..nodes import Permute
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseConsecutivePermutes(NodewiseOptimizationPass):
    """Fuse two consecutive permutes."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (permute := Permute.specialize_from(node))
            and (
                children := [
                    child_permute
                    for user in node.users
                    if ((child_permute := Permute.specialize_from(user)) and child_permute.ndim == permute.ndim)
                ]
            )
        ):
            return {}
        results: dict[Node, NodewisePassResult] = {}
        graph = node.graph
        for child_permute in children:
            # e.g. (N, C, H, W)  -[0, 3, 1, 2]-> [N, W, C, H] -[0, 2, 1, 3]-> (N, C, W, H)
            # is equivalent to (N, C, H, W) -[0, 1, 3, 2]-> (N, C, W, H)
            dims = [permute.dims[child_permute.dims[i]] for i in range(permute.ndim)]
            with graph.inserting_after(child_node := child_permute.node):
                fused_permute = graph.call_function(torch.ops.aten.permute.default, (permute.this, dims))
                fused_permute.stack_trace = (
                    f"{child_node.stack_trace}, pass: {node} and {child_node} fused by {__name__}"
                )
            results[child_node] = ReplaceAllUses(by=fused_permute)
        return results
