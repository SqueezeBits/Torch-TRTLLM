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
from transformers.utils.quantization_config import QuantizationMethod

from ...types import DataType
from ..nodes import GetAttr
from ..subgraphs import Linear
from ..targets import WeightOnlyGroupwiseQuantMatmulPlugin, WeightOnlyQuantMatmulPlugin
from ..targets.weightonly_quantmatmul_plugin import WeightTypeId
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ReplaceMMByWoQGemmPlugin(NodewiseOptimizationPass):
    """Replace MM node by Plugin for weight-only quantization.

    Attributes:
        model_dtype (torch.dtype): The data type of the model
    """

    model_dtype: torch.dtype

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (linear := Linear.configure_from(node))
            and linear.activation_quantization is None
            and (dequantize := linear.weight_dequantize_node) is not None
            and (unpacked_weight := GetAttr.specialize_from(dequantize.qweight))
            and (scale := GetAttr.specialize_from(dequantize.scale))
            and (
                local_trtllm_quant_algo := inference_trtllm_quant_algo(
                    dequantize.target.bits,
                    dequantize.target.dtype,
                    hf_quant_method=dequantize.target.global_quant_config.hf_quant_method,
                )
            )
            in (
                QuantAlgo.W4A16,
                QuantAlgo.W4A16_GPTQ,
                QuantAlgo.W8A16_GPTQ,
                QuantAlgo.W4A16_AWQ,
                QuantAlgo.W8A16,
            )
        ):
            return {}

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
            per_group=dequantize.target.group_size is not None,
        )
        with node.graph.inserting_before(unpacked_weight.node):
            postprocessed_qweight = GetAttr.create(
                node.graph, f"{unpacked_weight.name}_postprocessed", postprocessed_qweight_tensor
            )

        plugin_inputs: list[Node] = [linear.input_node, postprocessed_qweight.node, scale.node]

        if zeros := GetAttr.specialize_from(dequantize.zeros) if dequantize.zeros is not None else None:
            postprocessed_zeros_tensor = postprocess_zeros_for_trtllm(
                zeros.tensor,
                dequantize.target.global_quant_config.hf_quant_method,
                scale=scale.tensor,
                model_dtype=self.model_dtype,
            )
            with node.graph.inserting_before(zeros.node):
                postprocessed_zeros = GetAttr.create(
                    node.graph, f"{zeros.name}_postprocessed", postprocessed_zeros_tensor
                )
                plugin_inputs.append(postprocessed_zeros.node)

        with node.graph.inserting_before(node):
            plugin: WeightOnlyQuantMatmulPlugin | WeightOnlyGroupwiseQuantMatmulPlugin
            if dequantize.target.group_size is not None:
                plugin = WeightOnlyGroupwiseQuantMatmulPlugin(
                    type_id=DataType(dequantize.target.dtype).to(trt.DataType),
                    quant_algo=get_weightonly_groupwise_quant_algo(
                        False,
                        zeros is not None,
                        False,
                        local_trtllm_quant_algo == QuantAlgo.W4A8_AWQ,
                        dequantize.target.bits == 8,
                    ),
                    group_size=dequantize.target.group_size,
                )
            else:
                assert len(plugin_inputs) == 3, "Zero point is not supported for weight-only quantization"
                plugin = WeightOnlyQuantMatmulPlugin(
                    type_id=DataType(dequantize.target.dtype).to(trt.DataType),
                    weight_type_id=WeightTypeId.INT8 if dequantize.target.bits == 8 else WeightTypeId.INT4,
                )

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


def postprocess_qweight_for_trtllm(
    qweight: torch.Tensor,
    bits: int,
    quant_method: QuantizationMethod,
    *,
    model_dtype: torch.dtype,
    per_group: bool = False,
) -> torch.Tensor:
    """Postprocess the quantized weight tensor for TensorRT-LLM.

    Args:
        qweight (torch.Tensor): The quantized weight tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizationMethod): The quantization method used
        model_dtype (torch.dtype): The model data type.
        per_group (bool): Whether the quantization is per group. Defaults to False.

    Returns:
        torch.Tensor: The postprocessed weight tensor for TensorRT-LLM
    """
    assert bits in (4, 8), "Only 4-bit or 8-bit quantization is supported for Weight-Only Quantization of TensorRT-LLM"
    if quant_method in (QuantizationMethod.AWQ, QuantizationMethod.GPTQ, QuantizationMethod.COMPRESSED_TENSORS):
        assert qweight.dtype in (torch.uint8, torch.int8), f"Unsupported tensor dtype: {qweight.dtype=}"
        if bits == 4:
            qweight = (qweight[:, 1::2] * 16 + qweight[:, ::2]).view(torch.int8)
        weight_dtype = torch.int8 if bits == 8 else torch.quint4x2
        qweight = torch.ops.trtllm.preprocess_weights_for_mixed_gemm(qweight, weight_dtype, torch.float16)
        if per_group:
            qweight = qweight.view(model_dtype)
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    return qweight


def postprocess_zeros_for_trtllm(
    zeros: torch.Tensor,
    quant_method: QuantizationMethod,
    *,
    scale: torch.Tensor,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Postprocess the quantized zero point tensor for TensorRT-LLM.

    Args:
        zeros (torch.Tensor): The quantized zero point tensor
        quant_method (QuantizationMethod): The quantization method used
        scale (torch.Tensor): The scale tensor
        model_dtype (torch.dtype): The model data type

    Returns:
        torch.Tensor: The postprocessed zero point tensor for TensorRT-LLM
    """
    if quant_method in (QuantizationMethod.AWQ, QuantizationMethod.COMPRESSED_TENSORS, QuantizationMethod.GPTQ):
        zeros = zeros * scale
        zeros = zeros.to(model_dtype)
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    return zeros


def inference_trtllm_quant_algo(
    bits: int, compute_dtype: torch.dtype, *, hf_quant_method: QuantizationMethod
) -> QuantAlgo:
    """Infer the quantization algorithm for TensorRT-LLM .

    Args:
        bits (int): The number of bits used for quantization
        compute_dtype (torch.dtype): The compute data type
        hf_quant_method (QuantizationMethod): The quantization method used by the Hugging Face model

    Returns:
        QuantAlgo: The quantization algorithm for TensorRT-LLM
    """
    assert bits in (4, 8), "Only 4-bit and 8-bit quantization is supported for TensorRT-LLM"
    if hf_quant_method in (QuantizationMethod.AWQ, QuantizationMethod.COMPRESSED_TENSORS, QuantizationMethod.GPTQ):
        quant_algo: str = f"W{bits}A{compute_dtype.itemsize * 8}"
        if hf_quant_method == QuantizationMethod.GPTQ:
            quant_algo = f"{quant_algo}_GPTQ"
        elif hf_quant_method == QuantizationMethod.AWQ:
            quant_algo = f"{quant_algo}_AWQ"
    else:
        raise RuntimeError(f"Unsupported quantization method: {hf_quant_method}")

    assert quant_algo in QuantAlgo, f"Unsupported quantization algorithm: {quant_algo}"
    return QuantAlgo[quant_algo]
