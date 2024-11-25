import torch
from torch.fx import Node

from ..nodes import MM
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class CastFP16MMToFP32(NodewiseOptimizationPass):
    """Prepend input FP32-castings and append output FP16-casting for a FP16 matmul node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (mm := MM.specialize_from(node))
            and (mm_output := get_tensor_metadata(mm.node))
            and mm_output.dtype == torch.float16
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            lhs_cast = graph.call_function(torch.ops.aten._to_copy.default, (mm.this,), {"dtype": torch.float32})
            if lhs := get_tensor_metadata(mm.this):
                populate_tensor_metadata(lhs_cast, lhs, dtype=torch.float32)
            rhs_cast = graph.call_function(torch.ops.aten._to_copy.default, (mm.other,), {"dtype": torch.float32})
            if rhs := get_tensor_metadata(mm.other):
                populate_tensor_metadata(rhs_cast, rhs, dtype=torch.float32)
            mm_fp32 = graph.call_function(mm.target, (lhs_cast, rhs_cast))
            populate_tensor_metadata(mm_fp32, mm_output, dtype=torch.float32)
            output_cast = graph.call_function(torch.ops.aten._to_copy.default, (mm_fp32,), {"dtype": torch.float16})
            populate_tensor_metadata(output_cast, mm_output)
        return {node: ReplaceAllUses(by=output_cast)}
