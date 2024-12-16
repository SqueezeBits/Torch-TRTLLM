from functools import reduce

import torch
from torch.fx import Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ..subgraphs import MMConst
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseMMConstSiblings(NodewiseOptimizationPass):
    """Fuse a group of constant matmul nodes sharing the same input tensor and reduction dimension size."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        graph = node.graph
        children = [child for user in node.users if (child := MMConst.configure_from(user))]
        if len(children) <= 1:
            return {}
        first_weight, *other_weights = (child.weight.parameter for child in children)
        if not (
            first_weight.ndim == 2
            and all(
                other_weight.ndim == 2 and first_weight.shape[0] == other_weight.shape[0]
                for other_weight in other_weights
            )
        ):
            return {}

        output_sizes = [user.weight.parameter.shape[1] for user in children]
        if 0 <= MATMUL_FUSION_MAX_OUTPUT_SIZE < sum(output_sizes):
            return {}

        slice_indices = cumulative_sums(output_sizes)
        fused_param_shape = (first_weight.shape[0], sum(output_sizes))
        fused_param_dtype = reduce(torch.promote_types, (child.weight.parameter.dtype for child in children))

        with graph.inserting_before(children[0].mm.node):
            # The "get_attr" nodes for existing parameters must be recreated
            # in order to avoid breaking topological orders of the nodes.
            get_attrs: list[Node] = []
            for child in children:
                get_attr = graph.get_attr(child.weight.target)
                get_attr.meta.update(child.weight.node.meta)
                get_attrs.append(get_attr)

            fused_param = graph.call_function(torch.ops.aten.cat.default, (get_attrs, 1))
            fused_param.stack_trace = f"{children[0].weight.node.stack_trace}, pass: fused by {type(self).__name__}"
            populate_tensor_metadata(
                fused_param,
                shape=fused_param_shape,
                dtype=fused_param_dtype,
            )

            fused_mm = graph.call_function(torch.ops.aten.mm.default, (node, fused_param))
            fused_mm.stack_trace = f"{children[0].mm.node.stack_trace}, pass: fused by {type(self).__name__}"
            if lhs_meta := get_tensor_metadata(node):
                populate_tensor_metadata(fused_mm, lhs_meta, shape=(*lhs_meta.shape[:-1], fused_param_shape[-1]))

            slices = [
                graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (fused_mm, 1, slice_indices[i], slice_indices[i + 1]),
                )
                for i in range(len(slice_indices) - 1)
            ]

        results: dict[Node, NodewisePassResult] = {}
        for child, s in zip(children, slices):
            if child_meta := get_tensor_metadata(child.mm.node):
                populate_tensor_metadata(s, child_meta)
            results[child.mm.node] = ReplaceAllUses(by=s)
        return results


def cumulative_sums(values: list[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums
