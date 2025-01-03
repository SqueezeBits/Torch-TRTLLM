from torch.fx import Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ..nodes import MM, Add, Cat, GetAttr, Slice
from ..subgraphs import Linear
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FuseLinearSiblings(NodewiseOptimizationPass):
    """Fuse a group of Linear subgraphs sharing the same input tensor."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        graph = node.graph
        linears = [linear for user in node.users if (linear := Linear.configure_from(user))]
        if len(linears) <= 1:
            return {}
        first_weight, *other_weights = (linear.weight.parameter for linear in linears)
        if not (
            first_weight.ndim == 2
            and all(
                other_weight.ndim == 2 and first_weight.shape[0] == other_weight.shape[0]
                for other_weight in other_weights
            )
        ):
            return {}

        output_sizes = [user.weight.parameter.shape[1] for user in linears]
        if 0 <= MATMUL_FUSION_MAX_OUTPUT_SIZE < sum(output_sizes):
            return {}

        slice_indices = cumulative_sums(output_sizes)

        with graph.inserting_before(linears[0].mm.node):
            # The "get_attr" nodes for existing parameters must be recreated
            # in order to avoid breaking topological orders of the nodes.
            get_attrs = [GetAttr.create(graph, linear.weight.target, linear.weight.parameter) for linear in linears]
            fused_param = Cat.create(graph, get_attrs, 1)
            fused_node: Node = MM.create(graph, node, fused_param).node
            nodes_to_replace = [linear.mm.node for linear in linears]
            inject_stack_trace_from(*nodes_to_replace, to=fused_node)

            if all(linear.has_bias for linear in linears):
                # The "get_attr" nodes for existing parameters must be recreated
                # in order to avoid breaking topological orders of the nodes.
                bias_get_attrs = [
                    GetAttr.create(graph, linear.bias.target, linear.bias.parameter) for linear in linears
                ]
                fused_bias_params = Cat.create(graph, bias_get_attrs)
                fused_node = Add.create(graph, fused_node, fused_bias_params).node
                nodes_to_replace = [linear.add.node for linear in linears]
                inject_stack_trace_from(*nodes_to_replace, to=fused_node)

            slices = [
                Slice.create(graph, fused_node, -1, slice_indices[i], slice_indices[i + 1])
                for i in range(len(slice_indices) - 1)
            ]

        results: dict[Node, NodewisePassResult] = {}
        for n, s in zip(nodes_to_replace, slices):
            inject_stack_trace_from(n, to=s)
            results[n] = ReplaceAllUses(by=s.node)
        return results


def cumulative_sums(values: list[int]) -> list[int]:
    sums = [0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums
