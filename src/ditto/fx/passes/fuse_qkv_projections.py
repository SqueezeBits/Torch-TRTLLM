from itertools import accumulate

from torch.fx import GraphModule, Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ...debug import save_for_debug
from ..nodes import MM, AddTensor, Cat, ScaledDotProductAttention, Slice
from ..subgraphs.linear import Linear, find_nearest_linear_projection
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FuseQKVProjections(NodewiseOptimizationPass):
    """Fuse input projections of an attention layer to a single Linear subgraph."""

    def preprocess(self, graph_module: GraphModule) -> None:
        super().preprocess(graph_module)
        save_for_debug("before_qkv_fusion", graph_module)

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

        if len(linear_layers := filter_unique_linear_nodes([q_proj, k_proj, v_proj])) == 1 or not are_weights_fusible(
            linear_layers
        ):
            return {}

        output_sizes = [user.weight_tensor.shape[1] for user in linear_layers]
        if 0 <= MATMUL_FUSION_MAX_OUTPUT_SIZE < sum(output_sizes):
            return {}

        with graph.inserting_before(linear_layers[0].mm.node):
            # The existing weight nodes must be recreated in order to avoid breaking topological orders.
            weight_nodes = [
                graph.create_node(
                    op=n.op,
                    target=n.target,
                    args=n.args,
                    kwargs=n.kwargs,
                    name=n.name,
                )
                for n in (linear.weight_node for linear in linear_layers)
            ]
            fused_param = Cat.create(graph, weight_nodes, 1)
            fused_node: Node = MM.create(graph, linear_layers[0].input_node, fused_param).node
            nodes_to_replace = [linear.mm.node for linear in linear_layers]
            inject_stack_trace_from(*nodes_to_replace, to=fused_node)

            if all(linear.bias_node is not None for linear in linear_layers):
                # The existing bias nodes must be recreated in order to avoid breaking topological orders.
                bias_nodes = [
                    graph.create_node(
                        op=n.op,
                        target=n.target,
                        args=n.args,
                        kwargs=n.kwargs,
                        name=n.name,
                    )
                    for linear in linear_layers
                    if (n := linear.bias_node) is not None
                ]
                fused_bias_params = Cat.create(graph, bias_nodes)
                fused_node = AddTensor.create(graph, fused_node, fused_bias_params).node
                nodes_to_replace = [linear.add.node for linear in linear_layers if linear.add is not None]
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


def filter_unique_linear_nodes(linear_layers: list[Linear]) -> list[Linear]:
    """Filter out duplicate linear layers.

    Args:
        linear_layers (list[Linear]): List of linear layers to filter

    Returns:
        list[Linear]: List containing only unique linear layers
    """
    return list({linear.mm.node: linear for linear in linear_layers}.values())


def are_weights_fusible(linear_layers: list[Linear]) -> bool:
    """Check if the weights of linear layers are fusible.

    Args:
        linear_layers (list[Linear]): A list of linear layers to check for fusibility

    Returns:
        bool: True if all linear layers have fusible weights, False otherwise
    """
    first_weight, *other_weights = (linear.weight_tensor for linear in linear_layers)
    return first_weight.ndim == 2 and all(
        other_weight.ndim == 2 and first_weight.shape[0] == other_weight.shape[0] for other_weight in other_weights
    )
