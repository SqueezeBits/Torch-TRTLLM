import torch
from torch.fx import Node

from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import MMConstNode


class CastMMConstToFP32(NodeWiseOptimizationPass):
    """Eliminate reshape whose target shape is identical to the input shape."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (mm := MMConstNode.specialize_from(node))
            and (mm_output := get_tensor_metadata(mm.node))
            and mm_output.dtype == torch.float16
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            lhs_cast = graph.call_function(torch.ops.aten._to_copy.default, (mm.lhs,), {"dtype": torch.float32})
            if lhs := get_tensor_metadata(mm.lhs):
                populate_tensor_metadata(lhs_cast, lhs, dtype=torch.float32)
            rhs_cast = graph.call_function(torch.ops.aten._to_copy.default, (mm.rhs,), {"dtype": torch.float32})
            if rhs := get_tensor_metadata(mm.rhs):
                populate_tensor_metadata(lhs_cast, rhs, dtype=torch.float32)
            mm_fp32 = graph.call_function(torch.ops.aten.mm.default, (lhs_cast, rhs_cast))
            populate_tensor_metadata(mm_fp32, mm_output, dtype=torch.float32)
            output_cast = graph.call_function(torch.ops.aten._to_copy.default, (mm_fp32,), {"dtype": torch.float16})
            populate_tensor_metadata(output_cast, mm_output)
        return {node: output_cast}
