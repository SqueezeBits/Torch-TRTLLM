import tensorrt as trt
import torch
from torch.fx import Node

from ...types import DataType
from ..nodes import MM
from ..targets import GemmPlugin
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class ReplaceMMByFakeGemmPlugin(NodewiseOptimizationPass):
    """Replace torch.ops.aten.mm.default by FakeGemmPlugin (required for trtllm)."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (mm := MM.specialize_from(node))
            and (mm_output := get_tensor_metadata(mm.node))
            and (this := get_tensor_metadata(mm.this))
            and len(this.shape) == 2
            and (other := get_tensor_metadata(mm.other))
            and len(other.shape) == 2
        ):
            return {}

        graph = node.graph
        # Note: the right-hand-side `mm.other` must be transposed before it is fed to GemmPlugin
        # for the correct functionality as GemmPlugin's functionality breaks when `transb=0` (don't know why ...)
        with graph.inserting_after(mm.other):
            other_t = graph.call_function(torch.ops.aten.permute.default, (mm.other, (1, 0)))
            populate_tensor_metadata(other_t, other, shape=(other.shape[1], other.shape[0]))

        fake_gemm_plugin = GemmPlugin(transb=1, type_id=DataType(mm_output.dtype).to(trt.DataType))
        with graph.inserting_before(node):
            output = graph.call_function(fake_gemm_plugin, (mm.this, other_t))
            populate_tensor_metadata(output, mm_output)

        return {node: ReplaceAllUses(by=output)}
