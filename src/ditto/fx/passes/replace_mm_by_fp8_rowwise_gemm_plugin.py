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
from tensorrt_llm.quantization import QuantMode
from torch.fx import Node

from ...quantization import QuantizeMode, QuantizeType
from ...types import DataType
from ..nodes import Dequantize, GetAttr, GetItem, Permute, RmsnormQuantization, ToCopy
from ..subgraphs import Linear
from ..targets import Fp8RowwiseGemmPlugin, QuantizePerTokenPlugin
from ..utils import get_val, name_generator
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ReplaceMMByFp8RowwiseGemmPlugin(NodewiseOptimizationPass):
    """Replace torch.ops.aten.mm.default by Fp8RowwiseGemmPlugin (required for trtllm).

    This pass must be run after ReplaceRmsNormByFp8RmsNormPlugin.

    Attributes:
        model_dtype (torch.dtype): Data type of the model
    """

    model_dtype: torch.dtype

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (linear := Linear.configure_from(node))
            and isinstance(this := get_val(linear.mm.this), torch.Tensor)
            and len(this.shape) == 2
            and isinstance(other := get_val(linear.mm.other), torch.Tensor)
            and len(other.shape) == 2
            and (activation_quantization := linear.activation_quantization) is not None
            and activation_quantization.bits == 8
            and activation_quantization.type == QuantizeType.FLOAT
            and activation_quantization.quant_mode == QuantizeMode.PER_TOKEN
            and activation_quantization.dynamic
            and (dequantize := Dequantize.specialize_from(linear.mm.other)) is not None
            and dequantize.qweight_tensor is not None
            and dequantize.qweight_tensor.dtype == torch.float8_e4m3fn
            and dequantize.scale_tensor is not None
            and dequantize.target.mode == QuantizeMode.PER_CHANNEL
            and (graph_module := node.graph.owning_module) is not None
        ):
            return {}

        clamp_name_gen = name_generator(graph_module, "fp8rowwise_mm_clamp_val")
        input_node = linear.input_node
        token_scale_node: Node | None = None
        if (
            isinstance(
                input_val := get_val(linear.reshape_in.this if linear.reshape_in is not None else linear.input_node),
                torch.Tensor,
            )
            and input_val.dtype != torch.float8_e4m3fn
        ):
            with node.graph.inserting_before(node):
                clamp_val = GetAttr.create(
                    node.graph, next(clamp_name_gen), torch.tensor([-1200.0, 1200.0], dtype=torch.float32)
                )
            with node.graph.inserting_before(node):
                quantize_per_token = QuantizePerTokenPlugin(
                    type_id=trt.fp8,
                    quant_mode=QuantMode.from_description(use_fp8_rowwise=True),
                    clamp_enabled=True,
                    sum_per_token=False,
                )
                quantize_per_token_node = node.graph.call_function(
                    quantize_per_token,
                    (
                        input_node,
                        clamp_val.node,
                    ),
                )
            with node.graph.inserting_before(node):
                input_node = GetItem.create(node.graph, quantize_per_token_node, 0).node
                token_scale_node = GetItem.create(node.graph, quantize_per_token_node, 1).node

        if token_scale_node is None:
            assert (
                linear.reshape_in is not None
                and isinstance(rmsnorm_node := linear.reshape_in.this.args[0], Node)
                and (rmsnorm_quantization := RmsnormQuantization.specialize_from(rmsnorm_node))
            )
            with node.graph.inserting_before(node):
                token_scale_node = GetItem.create(node.graph, rmsnorm_quantization.node, 1).node

        channel_scale_node = dequantize.scale
        if dequantize.scale_tensor.dtype != torch.float32:
            with node.graph.inserting_before(node):
                channel_scale_node = ToCopy.create(node.graph, channel_scale_node, dtype=torch.float32).node

        fp8_rowwise_gemm_plugin = Fp8RowwiseGemmPlugin(
            has_per_channel_scaling=True,
            has_per_token_scaling=True,
            type_id=DataType(self.model_dtype).to(trt.DataType),
        )
        with node.graph.inserting_before(node):
            permute_node = Permute.create(node.graph, dequantize.qweight, (1, 0)).node
            fp8_rowwise_gemm_plugin_node = node.graph.call_function(
                fp8_rowwise_gemm_plugin,
                (
                    input_node,
                    permute_node,
                    token_scale_node,
                    channel_scale_node,
                ),
            )
            propagate_metadata_from(linear.mm, to=fp8_rowwise_gemm_plugin_node)

        return {node: ReplaceAllUses(by=fp8_rowwise_gemm_plugin_node)}
