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

from ...quantization import GlobalQuantConfig, QuantizeMode, QuantizeType
from ...types import DataType
from ..nodes import GetAttr, GetItem, SqueezeDim
from ..subgraphs import RmsNormSubgraph
from ..targets import RmsnormQuantizationPlugin
from ..utils import get_val, name_generator
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class ReplaceRmsNormByFp8RmsNormPlugin(NodewiseOptimizationPass):
    """Replace normalization layers by RmsnormQuantization for FP8 precision (required for trtllm)."""

    model_dtype: torch.dtype = torch.float16
    global_quant_config: GlobalQuantConfig | None

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            self.global_quant_config is not None
            and len(self.global_quant_config.quant_configs) == 1
            and self.global_quant_config.quant_configs[0].input_quant_scheme is not None
            and self.global_quant_config.quant_configs[0].input_quant_scheme.bits == 8
            and self.global_quant_config.quant_configs[0].input_quant_scheme.type == QuantizeType.FLOAT
            and self.global_quant_config.quant_configs[0].input_quant_scheme.mode == QuantizeMode.PER_TOKEN
            and self.global_quant_config.quant_configs[0].input_quant_scheme.dynamic
            and (rmsnorm := RmsNormSubgraph.configure_from(node))
            and (graph_module := node.graph.owning_module) is not None
        ):
            return {}

        bias_name_gen = name_generator(graph_module, "rmsnorm_bias")
        clamp_name_gen = name_generator(graph_module, "rmsnorm_clamp_val")
        scale_name_gen = name_generator(graph_module, "rmsnorm_scale")

        weight = rmsnorm.weight
        with node.graph.inserting_before(rmsnorm.input_node):
            bias = (
                rmsnorm.bias
                if rmsnorm.bias is not None
                else GetAttr.create(
                    node.graph, next(bias_name_gen), torch.zeros(weight.tensor.shape, dtype=self.model_dtype)
                )
            )
            clamp_val = GetAttr.create(
                node.graph, next(clamp_name_gen), torch.tensor([-1200.0, 1200.0], dtype=torch.float32)
            )
            scale = GetAttr.create(node.graph, next(scale_name_gen), torch.ones((1,), dtype=torch.float32))

        rmsnorm_quantized_plugin = RmsnormQuantizationPlugin(
            eps=rmsnorm.eps,
            dyn_act_scaling=True,
            sum_per_token=False,
            clamp_enabled=True,
            quant_mode=QuantMode.from_description(use_fp8_rowwise=True),
            type_id=DataType(self.model_dtype).to(trt.DataType),
            out_type_id=trt.fp8,
        )
        with node.graph.inserting_before(node):
            assert isinstance(input_val := get_val(rmsnorm.input_node), torch.Tensor) and (
                len(input_val.shape) == 2 or (len(input_val.shape) == 3 and input_val.shape[0] == 1)
            )
            input_node = rmsnorm.input_node
            if len(input_val.shape) == 3 and input_val.shape[0] == 1:
                input_node = SqueezeDim.create(node.graph, input_node, 0).node

            plugin_node = node.graph.call_function(
                rmsnorm_quantized_plugin,
                (
                    input_node,
                    weight.node,
                    bias.node,
                    scale.node,
                    clamp_val.node,
                ),
            )
            hidden_state_node = GetItem.create(node.graph, plugin_node, 0).node

        return {node: ReplaceAllUses(by=hidden_state_node)}
