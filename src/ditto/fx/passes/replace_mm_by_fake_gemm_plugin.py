import torch
from torch.fx import Node

from ..nodes import MM
from ..targets import GemmPlugin
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses
from ...types import map_torch_to_trt_dtype


class ReplaceMMByFakeGemmPlugin(NodewiseOptimizationPass):
    """Replace torch.ops.aten.mm.default by FakeGemmPlugin (required for trtllm)."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (mm := MM.specialize_from(node))
            and (mm_output := get_tensor_metadata(mm.node))
            and (this := get_tensor_metadata(mm.this))
            and (other := get_tensor_metadata(mm.other))
        ):
            return {}
        assert len(this.shape) == 2
        assert len(other.shape) == 2

        graph = node.graph
        # Note: It assume that a shape of the matrix weight is `k x n`.
        # But, it should be transposed for a functionality from `k x n` to `n x k`.
        with graph.inserting_after(mm.other):
            other_t = graph.call_function(torch.ops.aten.permute.default, (mm.other, (1, 0)))
            populate_tensor_metadata(other_t, other, shape=(other.shape[1], other.shape[0]))

        fake_gemm_plugin = GemmPlugin(transb=1, type_id=map_torch_to_trt_dtype(mm_output.dtype))
        with graph.inserting_before(node):
            output = graph.call_function(fake_gemm_plugin, (mm.this, other_t))
            populate_tensor_metadata(output, mm_output)

        return {node: ReplaceAllUses(by=output)}
