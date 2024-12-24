from torch.fx import Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ..nodes import MM, Cat, GetAttr, Slice
from ..subgraphs import MMConst
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


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

        with graph.inserting_before(children[0].mm.node):
            # The "get_attr" nodes for existing parameters must be recreated
            # in order to avoid breaking topological orders of the nodes.
            get_attrs = [GetAttr.create(graph, child.weight.target, child.weight.parameter) for child in children]
            fused_param = Cat.create(graph, get_attrs, 1)
            fused_mm = MM.create(graph, node, fused_param)
            inject_stack_trace_from(children[0].mm, to=fused_mm, fusing=[child.mm for child in children])
            slices = [
                Slice.create(graph, fused_mm, 1, slice_indices[i], slice_indices[i + 1])
                for i in range(len(slice_indices) - 1)
            ]
        for child, s in zip(children, slices):
            inject_stack_trace_from(child.mm, to=s)
        return {child.mm.node: ReplaceAllUses(by=s.node) for child, s in zip(children, slices)}


def cumulative_sums(values: list[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums
