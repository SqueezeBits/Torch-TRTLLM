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

from ...quantization import QuantizeMode
from ...types import DataType
from ..nodes import Dequantize, GetAttr, Permute
from ..subgraphs import Linear
from ..targets import GemmPlugin, Quantizer
from ..utils import attr_name_generator, get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ReplaceMMByFp8GemmPlugin(NodewiseOptimizationPass):
    """Replace torch.ops.aten.mm.default by GemmPlugin for FP8 precision (required for trtllm)."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (linear := Linear.configure_from(node))
            and (this := get_tensor_metadata(linear.mm.this))
            and len(this.shape) == 2
            and (other := get_tensor_metadata(linear.mm.other))
            and len(other.shape) == 2
            and (activation_quantization := linear.activation_quantization) is not None
            and activation_quantization.quant_mode == QuantizeMode.PER_TENSOR
            and activation_quantization.scale is not None
            and (dequantize := Dequantize.specialize_from(linear.mm.other)) is not None
            and dequantize.input_tensor is not None
            and dequantize.input_tensor.dtype == torch.float8_e4m3fn
            and dequantize.scale_tensor is not None
            and dequantize.quantize_mode == QuantizeMode.PER_TENSOR
            and (graph_module := node.graph.owning_module) is not None
        ):
            return {}

        name_gen = attr_name_generator(graph_module, "activation_quant_scale")

        with node.graph.inserting_before(node):
            act_scale_attr = GetAttr.create(node.graph, next(name_gen), activation_quantization.scale)
            quantize = node.graph.call_function(
                Quantizer(),
                (
                    linear.input_node,
                    act_scale_attr.node,
                    dequantize.input_tensor.dtype,
                ),
            )

        gemm_plugin = GemmPlugin(
            transb=1,
            type_id=DataType(dequantize.output_dtype).to(trt.DataType),
            use_fp8=1,
            alpha=(activation_quantization.scale * dequantize.scale_tensor).item(),
        )
        with node.graph.inserting_before(node):
            permute = Permute.create(node.graph, dequantize.x, (1, 0))
            plugin_node = node.graph.call_function(gemm_plugin, (quantize, permute.node))
            propagate_metadata_from(linear.mm, to=plugin_node)

        return {node: ReplaceAllUses(by=plugin_node)}
