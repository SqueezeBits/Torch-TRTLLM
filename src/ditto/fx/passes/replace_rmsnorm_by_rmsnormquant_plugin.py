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
from tensorrt_llm.functional import QuantMode
from torch.fx import Node

from ...quantization import GlobalQuantConfig, QuantizeType
from ...types import DataType
from ..nodes import GetAttr, GetItem, MulTensorTensor, SqueezeDim
from ..subgraphs import Linear, RmsNormSubgraph
from ..targets import RmsnormQuantizationPlugin
from ..utils import attr_name_generator, find_nearest, get_val
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class ReplaceRmsNormByRmsNormQuantPlugin(NodewiseOptimizationPass):
    """Replace normalization layers by RmsnormQuantization for FP8 precision (required for trtllm).

    Attributes:
        model_dtype (torch.dtype): Data type of the model
        global_quant_config (GlobalQuantConfig | None): Global quantization configuration
    """

    model_dtype: torch.dtype
    global_quant_config: GlobalQuantConfig | None

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            self.global_quant_config is not None
            and len(self.global_quant_config.quant_configs) == 1
            and self.global_quant_config.quant_configs[0].input_quant_scheme is not None
            and (rmsnorm := RmsNormSubgraph.configure_from(node))
            and (
                nearest_linear := find_nearest(Linear, node, follow_parent=False, follow_first_only=False, max_depth=10)
            )
            and (activation_quantization := nearest_linear.activation_quantization) is not None
            and activation_quantization.bits == 8
            and activation_quantization.type in (QuantizeType.FLOAT, QuantizeType.INT)
            and (
                (activation_quantization.type == QuantizeType.FLOAT and activation_quantization.dynamic)
                or (
                    activation_quantization.type == QuantizeType.INT
                    and (
                        activation_quantization.dynamic
                        or ((scale_tensor := activation_quantization.scale) is not None)
                        and scale_tensor.ndim in (0, 1, 2)
                    )
                )
            )
            and (lm_head := Linear.find_last(node.graph))
            and nearest_linear != lm_head
            and (graph_module := node.graph.owning_module) is not None
        ):
            return {}

        weight_node: Node = rmsnorm.weight.node
        if activation_quantization.smoother is not None:
            with node.graph.inserting_before(rmsnorm.weight.node):
                smoother = GetAttr.create(
                    node.graph, next(attr_name_generator(graph_module, "smoother")), activation_quantization.smoother
                )
            with node.graph.inserting_after(rmsnorm.weight.node):
                weight_node = MulTensorTensor.create(node.graph, rmsnorm.weight, smoother).node

        with node.graph.inserting_before(rmsnorm.input_node):
            bias = (
                rmsnorm.bias
                if rmsnorm.bias is not None
                else GetAttr.create(
                    node.graph,
                    next(attr_name_generator(graph_module, "rmsnorm_bias")),
                    torch.zeros(rmsnorm.weight.tensor.shape, dtype=self.model_dtype),
                )
            )
            if activation_quantization.type == QuantizeType.FLOAT:
                clamp_val = GetAttr.create(
                    node.graph,
                    next(attr_name_generator(graph_module, "rmsnorm_clamp_val")),
                    torch.tensor([-1200.0, 1200.0], dtype=torch.float32),
                )
            scale_name_gen = attr_name_generator(graph_module, "rmsnorm_scale")
            if activation_quantization.dynamic:
                scale = GetAttr.create(node.graph, next(scale_name_gen), torch.ones((1,), dtype=torch.float32))
            else:
                assert (scale_tensor := activation_quantization.scale) is not None, "scale tensor is required"
                scale = GetAttr.create(node.graph, next(scale_name_gen), scale_tensor.to(torch.float32) * 127.0)

        rmsnorm_quantized_plugin = RmsnormQuantizationPlugin(
            eps=rmsnorm.eps,
            dyn_act_scaling=activation_quantization.dynamic,
            sum_per_token=False,
            clamp_enabled=activation_quantization.type == QuantizeType.FLOAT,
            quant_mode=QuantMode.from_description(use_fp8_rowwise=True)
            if activation_quantization.type == QuantizeType.FLOAT
            else QuantMode.use_smooth_quant(per_token=True),
            type_id=DataType(self.model_dtype).to(trt.DataType),
            out_type_id=trt.fp8 if activation_quantization.type == QuantizeType.FLOAT else trt.int8,
        )
        with node.graph.inserting_before(node):
            assert isinstance(input_val := get_val(rmsnorm.input_node), torch.Tensor) and (
                len(input_val.shape) == 2 or (len(input_val.shape) == 3 and input_val.shape[0] == 1)
            )
            input_node = rmsnorm.input_node
            if len(input_val.shape) == 3 and input_val.shape[0] == 1:
                input_node = SqueezeDim.create(node.graph, input_node, 0).node

            input_nodes = [input_node, weight_node, bias.node, scale.node]
            if self.global_quant_config.quant_configs[0].input_quant_scheme.type == QuantizeType.FLOAT:
                input_nodes.append(clamp_val.node)

            plugin_node = node.graph.call_function(
                rmsnorm_quantized_plugin,
                tuple(input_nodes),
            )
            hidden_state_node = (
                GetItem.create(node.graph, plugin_node, 0).node
                if rmsnorm_quantized_plugin.dyn_act_scaling
                else plugin_node
            )

        return {node: ReplaceAllUses(by=hidden_state_node)}
