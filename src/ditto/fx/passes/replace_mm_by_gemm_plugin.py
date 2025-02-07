# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorrt as trt
import torch
from torch.fx import Node

from ...types import DataType
from ..nodes import MM, Permute
from ..targets import GemmPlugin
from ..utils import get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ReplaceMMByGemmPlugin(NodewiseOptimizationPass):
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
        gemm_plugin = GemmPlugin(transb=1, type_id=DataType(mm_output.dtype).to(trt.DataType))
        with graph.inserting_before(node):
            other_t = Permute.create(graph, mm.other, (1, 0)).node
            plugin_node = graph.call_function(gemm_plugin, (mm.this, other_t))
            propagate_metadata_from(mm, to=plugin_node)

        return {node: ReplaceAllUses(by=plugin_node)}
