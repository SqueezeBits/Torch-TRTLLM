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
from ..nodes import FakeQuantize, GetAttr, GetItem, MulTensorTensor, Permute, RmsnormQuantization, ToCopy
from ..subgraphs import Linear
from ..targets import QuantizePerTokenPlugin, QuantizeTensorPlugin, SmoothQuantGemmPlugin
from ..utils import attr_name_generator, get_val
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ReplaceMMBySmoothQuantGemmPlugin(NodewiseOptimizationPass):
    """Replace torch.ops.aten.mm.default by SmoothQuantGemmPlugin (required for trtllm).

    This pass must be run after ReplaceRmsNormByRmsNormQuantPlugin.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (linear := Linear.configure_from(node))
            and isinstance(this := get_val(linear.mm.this), torch.Tensor)
            and len(this.shape) == 2
            and isinstance(other := get_val(linear.mm.other), torch.Tensor)
            and len(other.shape) == 2
            and (activation_quantization := linear.activation_quantization) is not None
            and activation_quantization.bits == 8
            and activation_quantization.type == QuantizeType.INT
            and activation_quantization.quant_mode in (QuantizeMode.PER_TENSOR, QuantizeMode.PER_TOKEN)
            and (fake_quantize := FakeQuantize.specialize_from(linear.mm.other)) is not None
            and fake_quantize.input_tensor is not None
            and fake_quantize.input_tensor.dtype == torch.int8
            and fake_quantize.scale_tensor is not None
            and fake_quantize.quantize_mode in (QuantizeMode.PER_TENSOR, QuantizeMode.PER_CHANNEL)
            and (graph_module := node.graph.owning_module) is not None
        ):
            return {}

        input_node = linear.input_node
        input_scale_node: Node | None = None
        input_scale_name_gen = attr_name_generator(graph_module, "input_scale")
        if (
            isinstance(
                input_val := get_val(linear.reshape_in.this if linear.reshape_in is not None else linear.input_node),
                torch.Tensor,
            )
            and input_val.dtype != torch.int8
        ):
            assert activation_quantization.smoother is not None, "smoother is required"
            with node.graph.inserting_before(node):
                input_node = MulTensorTensor.create(
                    node.graph,
                    input_node,
                    GetAttr.create(
                        node.graph,
                        next(attr_name_generator(graph_module, "smoother")),
                        activation_quantization.smoother,
                    ),
                ).node

                if activation_quantization.dynamic:
                    quantize_node = node.graph.call_function(
                        QuantizePerTokenPlugin(
                            type_id=trt.int8,
                            quant_mode=QuantMode.use_smooth_quant(per_token=True),
                            clamp_enabled=False,
                            sum_per_token=False,
                        ),
                        (input_node,),
                    )
                    input_scale_node = GetItem.create(node.graph, quantize_node, 1).node
                    input_node = GetItem.create(node.graph, quantize_node, 0).node
                else:
                    assert (input_scale := activation_quantization.scale) is not None, "input scale is required"
                    input_scale_node = GetAttr.create(node.graph, next(input_scale_name_gen), 1 / input_scale).node
                    input_node = node.graph.call_function(
                        QuantizeTensorPlugin(),
                        (
                            input_node,
                            input_scale_node,
                        ),
                    )

        if input_scale_node is None:
            with node.graph.inserting_before(node):
                if activation_quantization.dynamic:
                    assert (
                        linear.reshape_in is not None
                        and isinstance(rmsnorm_node := linear.reshape_in.this.args[0], Node)
                        and (rmsnorm_quantization := RmsnormQuantization.specialize_from(rmsnorm_node))
                    )
                    input_scale_node = GetItem.create(node.graph, rmsnorm_quantization.node, 1).node
                else:
                    assert (input_scale := activation_quantization.scale) is not None, "input scale is required"
                    input_scale_node = GetAttr.create(node.graph, next(input_scale_name_gen), input_scale).node

        assert (weight_scale_node := fake_quantize.scale) is not None, "weight scale is required"
        if fake_quantize.scale_tensor.dtype != torch.float32:
            with node.graph.inserting_before(node):
                weight_scale_node = ToCopy.create(node.graph, weight_scale_node, dtype=torch.float32).node

        with node.graph.inserting_before(node):
            permute_node = Permute.create(node.graph, fake_quantize.x, (1, 0)).node
            smooth_quant_gemm_plugin_node = node.graph.call_function(
                SmoothQuantGemmPlugin(
                    has_per_channel_scaling=fake_quantize.scale_tensor.ndim == 2,
                    has_per_token_scaling=activation_quantization.dynamic,
                    type_id=DataType(fake_quantize.output_dtype).to(trt.DataType),
                ),
                (
                    input_node,
                    permute_node,
                    input_scale_node,
                    weight_scale_node,
                ),
            )
            propagate_metadata_from(linear.mm, to=smooth_quant_gemm_plugin_node)

        return {node: ReplaceAllUses(by=smooth_quant_gemm_plugin_node)}
