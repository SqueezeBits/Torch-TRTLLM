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

from typing import Any

import torch
from auto_gptq.nn_modules.qlinear import qlinear_cuda_old
from loguru import logger
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.functional import unpack_int32_into_int8
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.quantization_config import (
    AwqConfig,
    GPTQConfig,
    QuantizationConfigMixin,
    QuantizationMethod,
)
from typing_extensions import Self

from .types import StrictlyTyped


class GlobalQuantConfig(StrictlyTyped):
    """Global quantization configuration.

    Attributes:
        hf_quant_method (QuantizationMethod): The quantization method used by the Hugging Face model
        trtllm_quant_algo (QuantAlgo): The quantization algorithm used by TRT-LLM
        bits (int): The number of bits used for quantization
        trtllm_kv_cache_quant_algo (QuantAlgo | None): The quantization algorithm used by TRT-LLM for the KV cache.
            Defaults to None.
        symmetric (bool | None): Whether the quantization is symmetric. Defaults to None.
        group_size (int | None): The size of the quantization group. Defaults to None.
        has_zero_point (bool): Whether the quantization uses a zero point. Defaults to False.
    """

    hf_quant_method: QuantizationMethod
    trtllm_quant_algo: QuantAlgo
    bits: int
    trtllm_kv_cache_quant_algo: QuantAlgo | None = None
    symmetric: bool | None = None
    group_size: int | None = None
    has_zero_point: bool = False

    @classmethod
    def create_from(cls, pretrained_config: Any) -> Self | None:
        """Create a GlobalQuantConfig from a pretrained config.

        Args:
            pretrained_config (Any): The pretrained config

        Returns:
            Self | None: The created GlobalQuantConfig or None if no quantization config is found
        """
        if not (
            isinstance(pretrained_config, PretrainedConfig)
            and (quantization_config := getattr(pretrained_config, "quantization_config", None)) is not None
            and isinstance(quantization_config, QuantizationConfigMixin)
        ):
            return None

        logger.info("Quantization config is found in the pretrained config")
        if isinstance(quantization_config, GPTQConfig):
            if quantization_config.bits not in (4, 8):
                raise ValueError(f"Unsupported GPTQ bits: {quantization_config.bits=}")
            return cls(
                hf_quant_method=quantization_config.quant_method,
                trtllm_quant_algo=QuantAlgo.W4A16_GPTQ if quantization_config.bits == 4 else QuantAlgo.W8A16_GPTQ,
                bits=quantization_config.bits,
                symmetric=quantization_config.sym,
                group_size=quantization_config.group_size,
            )
        if isinstance(quantization_config, AwqConfig):
            if quantization_config.bits not in (4, 8):
                raise ValueError(f"Unsupported AWQ bits: {quantization_config.bits=}")
            return cls(
                hf_quant_method=quantization_config.quant_method,
                trtllm_quant_algo=QuantAlgo.W4A16_AWQ if quantization_config.bits == 4 else QuantAlgo.W8A16,
                bits=quantization_config.bits,
                symmetric=None,
                group_size=quantization_config.group_size,
                has_zero_point=quantization_config.zero_point,
            )

        raise RuntimeError(f"Unsupported quantization algorithm: {quantization_config}")


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
    quant_algo: str = f"W{bits}A{compute_dtype.itemsize * 8}"
    if hf_quant_method == QuantizationMethod.GPTQ:
        quant_algo = f"{quant_algo}_GPTQ"
    elif hf_quant_method == QuantizationMethod.AWQ:
        quant_algo = f"{quant_algo}_AWQ"
    else:
        raise RuntimeError(f"Unsupported quantization method: {hf_quant_method}")

    assert quant_algo in QuantAlgo, f"Unsupported quantization algorithm: {quant_algo}"
    return QuantAlgo[quant_algo]


def resolve_qlinear_device_map(model: PreTrainedModel) -> None:
    """Resolve the device map for the QuantLinear module.

    Args:
        model (PreTrainedModel): The model to resolve the device map for
    """
    # Note: This is a temporary solution to resolve PendingUnbackedSymbolNotFound error during the fake propagation.
    for _, module in model.named_modules():
        if isinstance(module, qlinear_cuda_old.QuantLinear):
            if module.wf.device != module.qzeros.device:
                module.wf = module.wf.to(module.qzeros.device)


def unpack_qweight(tensor: torch.Tensor, bits: int, quant_method: QuantizationMethod) -> torch.Tensor:
    """Unpack the quantized weight tensor.

    Args:
        tensor (torch.Tensor): The quantized weight tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizationMethod): The quantization method used

    Returns:
        torch.Tensor: The unpacked weight tensor
    """
    if quant_method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ):
        assert bits in (4, 8), f"Unsupported GPTQ or AWQ bits: {bits=}"
        if bits == 4:
            tensor = unpack_int32_into_int8(
                tensor if quant_method is QuantizationMethod.AWQ else tensor.T,
                quant_method is QuantizationMethod.AWQ,
            )
            tensor = tensor.T if quant_method is QuantizationMethod.GPTQ else tensor
            tensor = tensor - 8
            tensor[tensor < 0] += 16
            tensor = tensor.view(torch.uint8).contiguous()
        else:
            tensor = (
                tensor.T.contiguous().view(torch.uint8).T.contiguous()
                if quant_method is QuantizationMethod.GPTQ
                else tensor.view(torch.uint8).contiguous()
            )
            tensor = (tensor - 128).to(torch.int8)
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    return tensor


def unpack_qzeros(tensor: torch.Tensor, bits: int, quant_method: QuantizationMethod) -> torch.Tensor:
    """Unpack the quantized zero point tensor.

    Args:
        tensor (torch.Tensor): The quantized zero point tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizationMethod): The quantization method used

    Returns:
        torch.Tensor: The unpacked zero point tensor
    """
    if quant_method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ):
        assert bits in (4, 8), f"Unsupported GPTQ or AWQ bits: {bits=}"
        if bits == 4:
            tensor = unpack_int32_into_int8(tensor, quant_method is QuantizationMethod.AWQ)
        else:
            tensor = tensor.view(torch.int8)
        tensor = -tensor + 2 ** (bits - 1) - 1 * (quant_method is QuantizationMethod.GPTQ)
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    return tensor


def postprocess_qweight_for_trtllm(
    qweight: torch.Tensor,
    bits: int,
    quant_method: QuantizationMethod,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Postprocess the quantized weight tensor for TensorRT-LLM.

    Args:
        qweight (torch.Tensor): The quantized weight tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizationMethod): The quantization method used
        model_dtype (torch.dtype): The model data type.

    Returns:
        torch.Tensor: The postprocessed weight tensor for TensorRT-LLM
    """
    if quant_method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ):
        assert qweight.dtype == torch.uint8, f"Unsupported tensor dtype: {qweight.dtype=}"
        assert bits in (4, 8), f"Unsupported GPTQ or AWQ bits: {bits=}"
        if bits == 4:
            qweight = (qweight[:, 1::2] * 16 + qweight[:, ::2]).view(torch.int8)
        weight_dtype = torch.int8 if bits == 8 else torch.quint4x2
        qweight = torch.ops.trtllm.preprocess_weights_for_mixed_gemm(qweight, weight_dtype, torch.float16).view(
            model_dtype
        )
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    return qweight


def postprocess_qzeros_for_trtllm(
    qzeros: torch.Tensor,
    bits: int,
    quant_method: QuantizationMethod,
    *,
    scale: torch.Tensor,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Postprocess the quantized zero point tensor for TensorRT-LLM.

    Args:
        qzeros (torch.Tensor): The quantized zero point tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizationMethod): The quantization method used
        scale (torch.Tensor): The scale tensor
        model_dtype (torch.dtype): The model data type

    Returns:
        torch.Tensor: The postprocessed zero point tensor for TensorRT-LLM
    """
    if quant_method in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ):
        assert bits in (4, 8), f"Unsupported GPTQ or AWQ bits: {bits=}"
        qzeros = qzeros * scale
        qzeros = qzeros.to(model_dtype)
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    return qzeros
