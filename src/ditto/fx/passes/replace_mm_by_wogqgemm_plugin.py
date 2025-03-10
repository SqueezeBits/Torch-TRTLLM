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
from loguru import logger
from tensorrt_llm.quantization import QuantAlgo
from torch.fx import Node

from ...quantization import (
    QuantizeMode,
    inference_trtllm_quant_algo,
    postprocess_qweight_for_trtllm,
    postprocess_qzeros_for_trtllm,
)
from ...types import DataType
from ..nodes import Dequantize, GetAttr
from ..subgraphs import Linear
from ..targets import WeightOnlyGroupwiseQuantMatmulPlugin, WeightOnlyQuantMatmulPlugin
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ReplaceMMByWoGQGemmPlugin(NodewiseOptimizationPass):
    """Replace MM node by plugin for weight-only groupwise quantization (required for trtllm).

    Attributes:
        model_dtype (torch.dtype): The data type of the model
    """

    model_dtype: torch.dtype

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (linear := Linear.configure_from(node))
            and linear.activation_quant_scale is None
            and linear.dequantize_node is not None
            and (dequantize := Dequantize.specialize_from(linear.dequantize_node))
            and dequantize.target.mode == QuantizeMode.PER_GROUP
            and (unpacked_weight := GetAttr.specialize_from(dequantize.qweight))
            and (scale := GetAttr.specialize_from(dequantize.scale))
        ):
            return {}

        local_trtllm_quant_algo = inference_trtllm_quant_algo(
            dequantize.target.bits,
            dequantize.target.dtype,
            hf_quant_method=dequantize.target.global_quant_config.hf_quant_method,
        )
        if dequantize.target.global_quant_config.trtllm_quant_algo != local_trtllm_quant_algo:
            logger.warning(
                f"The quantization algorithm of the dequantize node ({local_trtllm_quant_algo}) is different from "
                f"the global quantization algorithm ({dequantize.target.global_quant_config.trtllm_quant_algo}). "
                f"This is not expected and may lead to incorrect behavior."
            )

        postprocessed_qweight_tensor = postprocess_qweight_for_trtllm(
            unpacked_weight.tensor,
            dequantize.target.bits,
            dequantize.target.global_quant_config.hf_quant_method,
            model_dtype=self.model_dtype,
        )
        with node.graph.inserting_before(unpacked_weight.node):
            postprocessed_qweight = GetAttr.create(node.graph, unpacked_weight.name, postprocessed_qweight_tensor)

        plugin_inputs: list[Node] = [linear.input_node, postprocessed_qweight.node, scale.node]

        qzeros = GetAttr.specialize_from(dequantize.zeros) if dequantize.zeros is not None else None
        if qzeros is not None:
            postprocessed_qzeros_tensor = postprocess_qzeros_for_trtllm(
                qzeros.tensor,
                dequantize.target.bits,
                dequantize.target.global_quant_config.hf_quant_method,
                scale=scale.tensor,
                model_dtype=self.model_dtype,
            )
            with node.graph.inserting_before(qzeros.node):
                postprocessed_qzeros = GetAttr.create(node.graph, qzeros.name, postprocessed_qzeros_tensor)
                plugin_inputs.append(postprocessed_qzeros.node)

        with node.graph.inserting_before(node):
            if local_trtllm_quant_algo in (
                QuantAlgo.W4A16,
                QuantAlgo.W4A16_GPTQ,
                QuantAlgo.W8A16_GPTQ,
                QuantAlgo.W4A16_AWQ,
                QuantAlgo.W8A16,
            ):
                if dequantize.target.group_size is not None:
                    plugin = WeightOnlyGroupwiseQuantMatmulPlugin(
                        type_id=DataType(dequantize.target.dtype).to(trt.DataType),
                        quant_algo=get_weightonly_groupwise_quant_algo(
                            False,
                            qzeros is not None,
                            False,
                            local_trtllm_quant_algo == QuantAlgo.W4A8_AWQ,
                            unpacked_weight.tensor.dtype == torch.int8,
                        ),
                        group_size=dequantize.target.group_size,
                    )
                else:
                    plugin = WeightOnlyQuantMatmulPlugin(
                        type_id=DataType(dequantize.target.dtype).to(trt.DataType),
                        weight_type_id=DataType(postprocessed_qweight_tensor.dtype).to(trt.DataType),
                    )
            else:
                raise ValueError(f"Wrong quantization algorithm {local_trtllm_quant_algo}.")

            plugin_node = node.graph.call_function(plugin, tuple(plugin_inputs))
            propagate_metadata_from(linear.mm, to=plugin_node)

        return {node: ReplaceAllUses(by=plugin_node)}


def get_weightonly_groupwise_quant_algo(
    has_bias: bool, has_zeros: bool, has_pre_quant_scale: bool, is_w4a8_awq: bool, is_int8_weight: bool
) -> int:
    """Get the weight-only groupwise quantization algorithm type based on configuration flags.

    Args:
        has_bias (bool): Whether bias is present
        has_zeros (bool): Whether zero points are present
        has_pre_quant_scale (bool): Whether pre-quantization scale is present
        is_w4a8_awq (bool): Whether using W4A8 AWQ
        is_int8_weight (bool): Whether weights are int8

    Returns:
        int: The algorithm type encoded as a bit flag combination
    """
    flags = [has_bias, has_zeros, has_pre_quant_scale, is_w4a8_awq, is_int8_weight]
    return sum((flag << i) for i, flag in enumerate(flags))
