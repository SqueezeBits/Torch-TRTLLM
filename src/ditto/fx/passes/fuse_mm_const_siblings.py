from torch.fx import Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ..nodes import MM, Add, Cat, GetAttr, Slice
from ..subgraphs import MMConst
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FuseMMConstSiblings(NodewiseOptimizationPass):
    """Fuse a group of constant matmul nodes sharing the same input tensor and reduction dimension size."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        graph = node.graph
        mm_consts = [mm_const for user in node.users if (mm_const := MMConst.configure_from(user))]
        if len(mm_consts) <= 1:
            return {}
        first_weight, *other_weights = (mm_const.weight.parameter for mm_const in mm_consts)
        if not (
            first_weight.ndim == 2
            and all(
                other_weight.ndim == 2 and first_weight.shape[0] == other_weight.shape[0]
                for other_weight in other_weights
            )
        ):
            return {}

        output_sizes = [user.weight.parameter.shape[1] for user in mm_consts]
        if 0 <= MATMUL_FUSION_MAX_OUTPUT_SIZE < sum(output_sizes):
            return {}

        slice_indices = cumulative_sums(output_sizes)

        with graph.inserting_before(mm_consts[0].mm.node):
            # The "get_attr" nodes for existing parameters must be recreated
            # in order to avoid breaking topological orders of the nodes.
            get_attrs = [
                GetAttr.create(graph, mm_const.weight.target, mm_const.weight.parameter) for mm_const in mm_consts
            ]
            fused_param = Cat.create(graph, get_attrs, 1)
            fused_node: Node = MM.create(graph, node, fused_param).node
            nodes_to_replace: list[Node] = [mm_const.mm.node for mm_const in mm_consts]
            inject_stack_trace_from(mm_consts[0].mm, to=fused_node, fusing=[mm_const.mm for mm_const in mm_consts])

            adds = [
                add
                for mm_const in mm_consts
                if (len(mm_const.mm.node.users) == 1 and (add := Add.specialize_from([*mm_const.mm.node.users][0])))
            ]
            biases = [
                get_attr
                for add in adds
                if (isinstance(add.other, Node) and (get_attr := GetAttr.specialize_from(add.other)))
            ]
            if len(biases) == len(adds) == len(mm_consts) and all(
                (
                    mm.mm.node == add.this
                    and len(bias.parameter.shape) == 1
                    and mm.weight.parameter.shape[-1] == bias.parameter.shape[-1]
                )
                for mm, add, bias in zip(mm_consts, adds, biases)
            ):
                # The "get_attr" nodes for existing parameters must be recreated
                # in order to avoid breaking topological orders of the nodes.
                bias_get_attrs = [GetAttr.create(graph, bias.target, bias.parameter) for bias in biases]
                fused_bias_params = Cat.create(graph, bias_get_attrs)
                fused_node = Add.create(graph, fused_node, fused_bias_params).node
                nodes_to_replace = [add.node for add in adds]
                inject_stack_trace_from(adds[0], to=fused_node, fusing=nodes_to_replace)

            slices = [
                Slice.create(graph, fused_node, -1, slice_indices[i], slice_indices[i + 1])
                for i in range(len(slice_indices) - 1)
            ]

        results: dict[Node, NodewisePassResult] = {}
        for mm_const, node_to_replace, s in zip(mm_consts, nodes_to_replace, slices):
            inject_stack_trace_from(mm_const.mm, to=s)
            results[node_to_replace] = ReplaceAllUses(by=s.node)
        return results


def cumulative_sums(values: list[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums
