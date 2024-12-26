from itertools import zip_longest

import torch
from torch.fx import Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ..nodes import MM, Add, Cat, GetAttr, Slice
from ..subgraphs import MMConst
from ..utils import get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FuseMMConstSiblings(NodewiseOptimizationPass):
    """Fuse a group of constant matmul nodes sharing the same input tensor and reduction dimension size."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        graph = node.graph
        mms = [mm for user in node.users if (mm := MMConst.configure_from(user))]
        if len(mms) <= 1:
            return {}
        first_weight, *other_weights = (mm.weight.parameter for mm in mms)
        if not (
            first_weight.ndim == 2
            and all(
                other_weight.ndim == 2 and first_weight.shape[0] == other_weight.shape[0]
                for other_weight in other_weights
            )
        ):
            return {}

        output_sizes = [user.weight.parameter.shape[1] for user in mms]
        if 0 <= MATMUL_FUSION_MAX_OUTPUT_SIZE < sum(output_sizes):
            return {}

        slice_indices = cumulative_sums(output_sizes)

        with graph.inserting_before(mms[0].mm.node):
            # The "get_attr" nodes for existing parameters must be recreated
            # in order to avoid breaking topological orders of the nodes.
            get_attrs = [GetAttr.create(graph, mm.weight.target, mm.weight.parameter) for mm in mms]
            fused_param = Cat.create(graph, get_attrs, 1)
            fused_mm = MM.create(graph, node, fused_param)
            inject_stack_trace_from(mms[0].mm, to=fused_mm, fusing=[mm.mm for mm in mms])

            adds = [
                add
                for mm in mms
                if len(mm.mm.node.users) == 1
                and (add := Add.specialize_from([*mm.mm.node.users][0]))
                and isinstance(add.other, Node)
            ]
            if len(adds) == len(mms) and all(
                mm.mm.node == add.this and len(bias.shape) == 1 and weight.shape[-1] == bias.shape[-1]
                for mm, add in zip(mms, adds)
                if (weight := get_tensor_metadata(mm.weight.node)) and (bias := get_tensor_metadata(add.other))  # type: ignore[arg-type]
            ):
                bias_get_attrs = [
                    GetAttr.create(
                        graph,
                        add.other.target,  # type: ignore[arg-type, union-attr]
                        GetAttr.specialize_from(add.other).parameter,  # type: ignore[arg-type, union-attr]
                    )
                    for add in adds
                ]
                fused_bias_params = Cat.create(graph, bias_get_attrs)
                fused_add = Add.create(graph, fused_mm, fused_bias_params)
                inject_stack_trace_from(adds[0], to=fused_add, fusing=adds)

                fused_mm = fused_add  # type: ignore[assignment]

            slices = [
                Slice.create(graph, fused_mm, -1, slice_indices[i], slice_indices[i + 1])
                for i in range(len(slice_indices) - 1)
            ]

        results: dict[Node, NodewisePassResult] = {}
        for mm, add, s in zip_longest(mms, adds, slices):
            inject_stack_trace_from(mm.mm, to=s)
            if fused_mm.target == torch.ops.aten.mm.default:
                results[mm.mm.node] = ReplaceAllUses(by=s.node)
            else:
                results[add.node] = ReplaceAllUses(by=s.node)
        return results


def cumulative_sums(values: list[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums
