from itertools import accumulate

from torch.fx import Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ..nodes import MM, Add, Cat, ScaledDotProductAttention, Slice
from ..subgraphs import Linear
from ..utils import get_ancestors_with_depth
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FuseQKVProjection(NodewiseOptimizationPass):
    """Fuse a group of Linear subgraphs sharing the same input tensor."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        graph = node.graph
        if not (
            (sdpa := ScaledDotProductAttention.specialize_from(node))
            and (q_proj := find_nearest_linear_projection(sdpa.query))
            and (k_proj := find_nearest_linear_projection(sdpa.key))
            and (v_proj := find_nearest_linear_projection(sdpa.value))
            and q_proj.input_node == k_proj.input_node == v_proj.input_node
        ):
            return {}

        if len(linears := filter_unique_linears([q_proj, k_proj, v_proj])) == 1 or not are_weights_fusible(linears):
            return {}

        output_sizes = [user.weight_tensor.shape[1] for user in linears]
        if 0 <= MATMUL_FUSION_MAX_OUTPUT_SIZE < sum(output_sizes):
            return {}

        with graph.inserting_before(linears[0].mm.node):
            # The existing weight nodes must be recreated in order to avoid breaking topological orders.
            weight_nodes = [
                graph.create_node(
                    op=n.op,
                    target=n.target,
                    args=n.args,
                    kwargs=n.kwargs,
                    name=n.name,
                )
                for n in (linear.weight_node for linear in linears)
            ]
            fused_param = Cat.create(graph, weight_nodes, 1)
            fused_node: Node = MM.create(graph, linears[0].input_node, fused_param).node
            nodes_to_replace = [linear.mm.node for linear in linears]
            inject_stack_trace_from(*nodes_to_replace, to=fused_node)

            if all(linear.bias_node is not None for linear in linears):
                # The existing bias nodes must be recreated in order to avoid breaking topological orders.
                bias_nodes = [
                    graph.create_node(
                        op=n.op,
                        target=n.target,
                        args=n.args,
                        kwargs=n.kwargs,
                        name=n.name,
                    )
                    for linear in linears
                    if (n := linear.bias_node) is not None
                ]
                fused_bias_params = Cat.create(graph, bias_nodes)
                fused_node = Add.create(graph, fused_node, fused_bias_params).node
                nodes_to_replace = [linear.add.node for linear in linears if linear.add is not None]
                inject_stack_trace_from(*nodes_to_replace, to=fused_node)

            slice_indices = [0, *accumulate(output_sizes)]
            slices = [
                Slice.create(graph, fused_node, -1, slice_indices[i], slice_indices[i + 1])
                for i in range(len(slice_indices) - 1)
            ]

        results: dict[Node, NodewisePassResult] = {}
        for n, s in zip(nodes_to_replace, slices):
            inject_stack_trace_from(n, to=s)
            results[n] = ReplaceAllUses(by=s.node)
        return results


def filter_unique_linears(linears: list[Linear]) -> list[Linear]:
    """Filter out duplicate linear layers.

    Args:
        linears (list[Linear]): List of linear layers to filter

    Returns:
        list[Linear]: List containing only unique linear layers
    """
    return list({linear.mm.node: linear for linear in linears}.values())


def are_weights_fusible(linears: list[Linear]) -> bool:
    """Check if the weights of linear layers are fusible.

    Args:
        linears (list[Linear]): A list of linear layers to check for fusibility

    Returns:
        bool: True if all linear layers have fusible weights, False otherwise
    """
    first_weight, *other_weights = (linear.weight_tensor for linear in linears)
    return first_weight.ndim == 2 and all(
        other_weight.ndim == 2 and first_weight.shape[0] == other_weight.shape[0] for other_weight in other_weights
    )


def find_nearest_linear_projection(x: Node) -> Linear | None:
    """Find the nearest Linear projection subgraph by traversing up the node's ancestors.

    Searches through all ancestor nodes and finds the Linear projection subgraph that is closest
    to the given node in terms of graph traversal depth. This is useful for identifying the
    linear transformation that most directly affects the node's computation.

    Args:
        x (Node): Starting node to search ancestors from

    Returns:
        The nearest Linear projection subgraph if one exists in the ancestors, None otherwise
    """
    if not (
        ancester_linear_subgraphs := {
            subgraph: depth
            for node, depth in get_ancestors_with_depth(x).items()
            if (subgraph := Linear.configure_from(node))
        }
    ):
        return None
    return min(ancester_linear_subgraphs, key=lambda subgraph: ancester_linear_subgraphs[subgraph])
