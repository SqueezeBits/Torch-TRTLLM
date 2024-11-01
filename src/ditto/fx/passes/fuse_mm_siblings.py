import operator

import torch
from torch.fx import Node

from ...config import MAX_FUSIBLE_MATMUL_OUT_SIZE
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import MMConstNode


class FuseMMSiblings(NodeWiseOptimizationPass):
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
        if sum(output_sizes) > MAX_FUSIBLE_MATMUL_OUT_SIZE:
            return {}

        graph = node.graph
        if not (graph_module := graph.owning_module):
            return {}

        graph_module.register_buffer(fused_weight_name, fused_weight)
        with graph.inserting_before(children[0].node):
            get_attr = graph.get_attr(fused_weight_name)
            populate_tensor_metadata(get_attr, fused_weight)
            fused_mm = graph.call_function(torch.ops.aten.mm.default, (node, get_attr))
            if lhs_meta := get_tensor_metadata(node):
                populate_tensor_metadata(fused_mm, lhs_meta, shape=(*lhs_meta.shape[:-1], fused_weight.shape[-1]))
            split = graph.call_function(torch.ops.aten.split.sizes, (fused_mm, output_sizes, -1))
            getitems = [graph.call_function(operator.getitem, (split, i)) for i in range(len(output_sizes))]
        replacements: dict[Node, Node] = {}
        for user, getitem in zip(children, getitems):
            if user_meta := get_tensor_metadata(user.node):
                populate_tensor_metadata(getitem, user_meta)
            replacements[user.node] = getitem
        return replacements
