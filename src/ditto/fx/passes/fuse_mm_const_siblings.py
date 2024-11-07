import torch
from torch.fx import Node

from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import MMConstNode


class FuseMMConstSiblings(NodeWiseOptimizationPass):
    """Fuse a group of constant matmul nodes sharing the same input tensor and reduction dimension size."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        children = [child_mm for user in node.users if (child_mm := MMConstNode.specialize_from(user))]
        if len(children) <= 1:
            return {}
        first_weight, *other_weights = (child_mm.weight for child_mm in children)
        if not (
            first_weight.ndim == 2
            and all(
                other_weight.ndim == 2 and first_weight.shape[0] == other_weight.shape[0]
                for other_weight in other_weights
            )
        ):
            return {}
        fused_weight = torch.cat([user.weight for user in children], dim=1).contiguous()
        fused_weight_name = "".join(user.weight_name for user in children)
        output_sizes = [user.weight.shape[1] for user in children]
        slice_indices = cumulative_sums(output_sizes)

        graph = node.graph
        if not (graph_module := graph.owning_module):
            return {}

        graph_module.register_parameter(fused_weight_name, torch.nn.Parameter(fused_weight, requires_grad=False))
        with graph.inserting_before(children[0].node):
            get_attr = graph.get_attr(fused_weight_name)
            populate_tensor_metadata(get_attr, fused_weight)
            fused_mm = graph.call_function(torch.ops.aten.mm.default, (node, get_attr))
            if lhs_meta := get_tensor_metadata(node):
                populate_tensor_metadata(fused_mm, lhs_meta, shape=(*lhs_meta.shape[:-1], fused_weight.shape[-1]))
            slices = [
                graph.call_function(torch.ops.aten.slice.Tensor, (fused_mm, 1, slice_indices[i], slice_indices[i + 1]))
                for i in range(len(slice_indices) - 1)
            ]
        replacements: dict[Node, Node] = {}
        for user, s in zip(children, slices):
            if user_meta := get_tensor_metadata(user.node):
                populate_tensor_metadata(s, user_meta)
            replacements[user.node] = s
        return replacements


def cumulative_sums(values: list[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums
