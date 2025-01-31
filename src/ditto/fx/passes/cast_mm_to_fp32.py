import torch
from torch.fx import Node

from ..nodes import MM, ToCopy
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class CastMMToFP32(NodewiseOptimizationPass):
    """Prepend input FP32-castings and append output type-casting for a specific-type matmul node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (mm := MM.specialize_from(node)) and (output_dtype := mm.output_dtype) and output_dtype != torch.float32
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            lhs_cast = ToCopy.create(graph, mm.this, dtype=torch.float32)
            rhs_cast = ToCopy.create(graph, mm.other, dtype=torch.float32)
            mm_fp32 = MM.create(graph, lhs_cast, rhs_cast)
            propagate_metadata_from(mm, to=mm_fp32)
            output_cast = ToCopy.create(graph, mm_fp32, dtype=mm.output_dtype)
        return {mm.node: ReplaceAllUses(by=output_cast.node)}
