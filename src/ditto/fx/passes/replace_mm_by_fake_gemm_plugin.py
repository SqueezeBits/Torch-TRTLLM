import tensorrt as trt
import torch
from torch.fx import Node

from ...types import DataType
from ..nodes import MM, Permute
from ..targets import GemmPlugin
from ..utils import get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class ReplaceMMByFakeGemmPlugin(NodewiseOptimizationPass):
    """Replace torch.ops.aten.mm.default by FakeGemmPlugin (required for trtllm)."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (mm := MM.specialize_from(node))
            and isinstance(mm_output := mm.output, torch.Tensor)
            and (this := get_tensor_metadata(mm.this))
            and len(this.shape) == 2
            and (other := get_tensor_metadata(mm.other))
            and len(other.shape) == 2
        ):
            return {}

        graph = node.graph
        # Note: the right-hand-side `mm.other` must be transposed before it is fed to GemmPlugin
        # for the correct functionality as GemmPlugin's functionality breaks when `transb=0` (don't know why ...)
        fake_gemm_plugin = GemmPlugin(transb=1, type_id=DataType(mm_output.dtype).to(trt.DataType))
        with graph.inserting_before(node):
            other_t = Permute.create(graph, mm.other, (1, 0)).node
            gemm_plugin = graph.call_function(fake_gemm_plugin, (mm.this, other_t))
            inject_stack_trace_from(mm, to=gemm_plugin)

        return {node: ReplaceAllUses(by=gemm_plugin)}
