import torch
from torch.fx import Node

from ...fake_targets import fake_transposed_mm
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import MMConstNode


class RewriteMMConstAsTransposedMM(NodeWiseOptimizationPass):
    """Rewrite activation-weight matmul as matmul with weight side transposed."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        graph = node.graph
        if not ((mm := MMConstNode.specialize_from(node)) and (graph_module := graph.owning_module)):
            return {}
        transposed_weight_name = f"{mm.weight_name}_transposed"
        transposed_weight = torch.nn.Parameter(mm.weight.data.permute(1, 0), requires_grad=mm.weight.requires_grad)
        graph_module.register_parameter(transposed_weight_name, transposed_weight)
        with graph.inserting_after(mm.rhs):
            populate_tensor_metadata(
                transposed_weight_node := graph.get_attr(transposed_weight_name),
                transposed_weight,
            )
        with graph.inserting_before(node):
            transposed_mm = graph.call_function(fake_transposed_mm, (mm.lhs, transposed_weight_node))
            if mm_output := get_tensor_metadata(mm.node):
                populate_tensor_metadata(transposed_mm, mm_output)
        return {node: transposed_mm}
