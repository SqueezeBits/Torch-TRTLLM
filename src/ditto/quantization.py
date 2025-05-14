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

from enum import Enum, auto

import torch
from auto_gptq.nn_modules.qlinear import qlinear_cuda_old
from awq.modules.linear.gemm import WQLinear_GEMM
from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy, QuantizationType
from compressed_tensors.quantization.lifecycle import KVCacheScaleType, is_attention_module
from loguru import logger
from modelopt.torch.quantization.model_calib import disable_pre_quant_scale_and_resmooth
from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear
from peft import PeftModel
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.functional import unpack_int32_into_int8
from torch._ops import OpOverload
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.quantization_config import (
    AwqConfig,
    CompressedTensorsConfig,
    GPTQConfig,
    QuantizationConfigMixin,
)
from typing_extensions import Self

from .literals import QuantizeMethod
from .torch.modules import QuantLinear
from .types import StrictlyTyped


class QuantizeType(Enum):
    """Quantization type.

    Attributes:
        FLOAT (auto): float quantization.
        INT (auto): int quantization.
    """

    FLOAT = auto()
    INT = auto()

    @classmethod
    def from_quant_type(cls, quant_type: QuantizationType) -> "QuantizeType":
        """Convert quantization type to Ditto quantization type.

        Args:
            quant_type (QuantizationType): The quantization type

        Returns:
            QuantizeType: The Ditto quantization type
        """
        if quant_type == QuantizationType.FLOAT:
            return cls.FLOAT
        if quant_type == QuantizationType.INT:
            return cls.INT

        raise NotImplementedError(f"Unsupported quantization type: {quant_type}")


class QuantizeMode(Enum):
    """Quantization mode.

    Attributes:
        PER_TENSOR (auto): per-tensor quantization.
        PER_GROUP (auto): per-group quantization.
        PER_CHANNEL (auto): per-channel quantization.
        PER_BLOCK (auto): per-block quantization.
        PER_TOKEN (auto): per-token quantization.
        UNKNOWN (auto): unknown quantization mode.
    """

    PER_TENSOR = auto()
    PER_GROUP = auto()
    PER_CHANNEL = auto()
    PER_BLOCK = auto()
    PER_TOKEN = auto()
    UNKNOWN = auto()

    @classmethod
    def from_quant_strategy(cls, quant_strategy: QuantizationStrategy) -> "QuantizeMode":
        """Convert quantization strategy to Ditto quantization mode.

        Args:
            quant_strategy (QuantizationStrategy): The quantization strategy

        Returns:
            QuantizeMode: The Ditto quantization mode
        """
        if quant_strategy == QuantizationStrategy.TENSOR:
            return cls.PER_TENSOR
        if quant_strategy == QuantizationStrategy.CHANNEL:
            return cls.PER_CHANNEL
        if quant_strategy == QuantizationStrategy.GROUP:
            return cls.PER_GROUP
        if quant_strategy == QuantizationStrategy.BLOCK:
            return cls.PER_BLOCK
        if quant_strategy == QuantizationStrategy.TOKEN:
            return cls.PER_TOKEN

        raise NotImplementedError(f"Unsupported quantization strategy: {quant_strategy}")


class QuantScheme(StrictlyTyped):
    """Quantization scheme.

    Attributes:
        bits (int): The number of bits used for quantization.
        mode (QuantizeMode): The quantization mode.
        type (QuantizeType | None): The type of the quantization.
        group_size (int | None): The size of the quantization group.
        has_zero_point (bool): Whether the quantization uses a zero point.
        dynamic (bool): Whether the quantization is dynamic.
    """

    bits: int
    mode: QuantizeMode
    type: QuantizeType | None = None
    group_size: int | None = None
    has_zero_point: bool = False
    dynamic: bool = False


class TargetQuantConfig(StrictlyTyped):
    """Target quantization config.

    Attributes:
        target (OpOverload): The target operator.
        input_quant_scheme (QuantScheme | None): The input quantization scheme.
        weight_quant_scheme (QuantScheme | None): The weight quantization scheme.
    """

    target: OpOverload
    input_quant_scheme: QuantScheme | None
    weight_quant_scheme: QuantScheme | None


class GlobalQuantConfig(StrictlyTyped):
    """Global quantization configuration.

    Attributes:
        quant_method (QuantizeMethod): The quantization method used by the Hugging Face model
        trtllm_quant_algo (QuantAlgo): The quantization algorithm used by TRT-LLM
        trtllm_kv_cache_quant_algo (QuantAlgo): The quantization algorithm used by TRT-LLM for the KV cache.
            Defaults to QuantAlgo.NO_QUANT.
        clamp_val (list[float] | None): The clamp values for the quantization. Defaults to None.
        quant_configs (list[TargetQuantConfig]): The quantization schemes for the target operators.
    """

    quant_method: QuantizeMethod
    trtllm_quant_algo: QuantAlgo
    trtllm_kv_cache_quant_algo: QuantAlgo = QuantAlgo.NO_QUANT
    clamp_val: list[float] | None = None
    quant_configs: list[TargetQuantConfig] = []

    @classmethod
    # pylint: disable-next=too-many-branches, too-many-statements
    def create_from(cls, model: PreTrainedModel | PeftModel) -> Self | None:
        """Create a GlobalQuantConfig from a pretrained config.

        Args:
            model (PreTrainedModel | PeftModel): The model to create the GlobalQuantConfig from

        Returns:
            Self | None: The created GlobalQuantConfig or None if no quantization config is found
        """
        input_quant_scheme: QuantScheme | None = None
        weight_quant_scheme: QuantScheme | None = None
        output_quant_scheme: QuantScheme | None = None

        if is_quantized_by_modelopt(model):
            input_quant_scheme, weight_quant_scheme, output_quant_scheme = get_modelopt_quantization_scheme(model)

            trtllm_quant_algo: QuantAlgo = QuantAlgo.NO_QUANT
            if input_quant_scheme is None:
                if weight_quant_scheme is not None:
                    if weight_quant_scheme.type == QuantizeType.INT:
                        trtllm_quant_algo = QuantAlgo.W8A16 if weight_quant_scheme.bits == 8 else QuantAlgo.W4A16
                    else:
                        raise NotImplementedError("Weight-only quantization for floating-point types is not supported")
            else:
                assert weight_quant_scheme is not None, "Weight quantization scheme is required"
                assert (
                    quantize_type := input_quant_scheme.type
                ) == weight_quant_scheme.type, "Input and weight quantization type must be the same"
                if quantize_type == QuantizeType.FLOAT:
                    if (input_quant_scheme.mode, weight_quant_scheme.mode) == (
                        QuantizeMode.PER_TENSOR,
                        QuantizeMode.PER_TENSOR,
                    ):
                        trtllm_quant_algo = QuantAlgo.FP8
                    elif (input_quant_scheme.mode, weight_quant_scheme.mode) == (
                        QuantizeMode.PER_TOKEN,
                        QuantizeMode.PER_CHANNEL,
                    ):
                        trtllm_quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
                    else:
                        raise NotImplementedError(
                            f"Unsupported quantization mode: {input_quant_scheme.mode}, {weight_quant_scheme.mode}"
                        )
                else:
                    raise NotImplementedError(f"Unsupported input/weight quantization type: {quantize_type=}")

            trtllm_kv_cache_quant_algo: QuantAlgo = QuantAlgo.NO_QUANT
            if output_quant_scheme is not None:
                logger.info("Output quantizations are found, which enables KV cache quantization")
                trtllm_kv_cache_quant_algo = (
                    QuantAlgo.INT8 if output_quant_scheme.type == QuantizeType.INT else QuantAlgo.FP8
                )

            return cls(
                quant_method="modelopt",
                trtllm_quant_algo=trtllm_quant_algo,
                trtllm_kv_cache_quant_algo=trtllm_kv_cache_quant_algo,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=input_quant_scheme,
                        weight_quant_scheme=weight_quant_scheme,
                    ),
                ],
            )

        if not (
            isinstance(pretrained_config := model.config, PretrainedConfig)
            and (quantization_config := getattr(pretrained_config, "quantization_config", None)) is not None
            and isinstance(quantization_config, QuantizationConfigMixin)
        ):
            return None

        logger.info("Quantization config is found in the pretrained config")
        if isinstance(quantization_config, GPTQConfig):
            if quantization_config.bits not in (4, 8):
                raise ValueError(f"Unsupported GPTQ bits: {quantization_config.bits=}")
            return cls(
                quant_method=quantization_config.quant_method.value,
                trtllm_quant_algo=QuantAlgo.W4A16_GPTQ if quantization_config.bits == 4 else QuantAlgo.W8A16_GPTQ,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=None,
                        weight_quant_scheme=QuantScheme(
                            bits=quantization_config.bits,
                            mode=QuantizeMode.PER_GROUP,
                            type=QuantizeType.INT,
                            group_size=quantization_config.group_size,
                        ),
                    ),
                ],
            )

        if isinstance(quantization_config, AwqConfig):
            if quantization_config.bits not in (4, 8):
                raise ValueError(f"Unsupported AWQ bits: {quantization_config.bits=}")
            return cls(
                quant_method=quantization_config.quant_method.value,
                trtllm_quant_algo=QuantAlgo.W4A16_AWQ if quantization_config.bits == 4 else QuantAlgo.W8A16,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=None,
                        weight_quant_scheme=QuantScheme(
                            bits=quantization_config.bits,
                            mode=QuantizeMode.PER_GROUP,
                            type=QuantizeType.INT,
                            group_size=quantization_config.group_size,
                            has_zero_point=quantization_config.zero_point,
                        ),
                    ),
                ],
            )

        if isinstance(quantization_config, CompressedTensorsConfig):
            input_quant_scheme, weight_quant_scheme, output_quant_scheme = get_compressed_tensors_quantization_scheme(
                quantization_config
            )

            trtllm_quant_algo = QuantAlgo.NO_QUANT
            if input_quant_scheme is None:
                if weight_quant_scheme is not None:
                    if weight_quant_scheme.type == QuantizeType.INT:
                        trtllm_quant_algo = QuantAlgo.W8A16 if weight_quant_scheme.bits == 8 else QuantAlgo.W4A16
                    else:
                        raise NotImplementedError(
                            f"Unsupported weight-only quantization type: {weight_quant_scheme.type=}"
                        )
            else:
                assert (
                    weight_quant_scheme is not None
                ), "Weight quantization scheme is required if input quantization scheme is provided"
                assert (
                    quantize_type := input_quant_scheme.type
                ) == weight_quant_scheme.type, "input and weight quantization type must be the same"
                if quantize_type == QuantizeType.FLOAT:
                    if (input_quant_scheme.mode, weight_quant_scheme.mode) == (
                        QuantizeMode.PER_TENSOR,
                        QuantizeMode.PER_TENSOR,
                    ):
                        trtllm_quant_algo = QuantAlgo.FP8
                    elif (input_quant_scheme.mode, weight_quant_scheme.mode) == (
                        QuantizeMode.PER_TOKEN,
                        QuantizeMode.PER_CHANNEL,
                    ):
                        trtllm_quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
                    else:
                        raise NotImplementedError(
                            f"Unsupported quantization mode: {input_quant_scheme.mode}, {weight_quant_scheme.mode}"
                        )
                else:
                    raise NotImplementedError(f"Unsupported input/weight quantization type: {quantize_type=}")

            if output_quant_scheme is not None:
                logger.info("Output quantizations are found, which enables KV cache quantization")
                trtllm_kv_cache_quant_algo = (
                    QuantAlgo.INT8 if output_quant_scheme.type == QuantizeType.INT else QuantAlgo.FP8
                )

            return cls(
                quant_method=quantization_config.quantization_config.quant_method,
                trtllm_quant_algo=trtllm_quant_algo,
                trtllm_kv_cache_quant_algo=trtllm_kv_cache_quant_algo,
                clamp_val=[-1200.0, 1200.0] if trtllm_quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN else None,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=input_quant_scheme,
                        weight_quant_scheme=weight_quant_scheme,
                    ),
                ],
            )

        raise RuntimeError(f"Unsupported quantization algorithm: {quantization_config}")


def update_kv_cache_scales(
    model: PreTrainedModel | PeftModel, quant_method: QuantizeMethod, kv_cache_quant_algo: QuantAlgo
) -> None:
    """Updating the KV cache scales into CompressedLinear modules.

    Args:
        model (PreTrainedModel | PeftModel): The model to update the KV cache scales
        quant_method (QuantizeMethod): The quantization method used
        kv_cache_quant_algo (QuantAlgo): The quantization algorithm used for the KV cache
    """
    if quant_method == "compressed-tensors" and kv_cache_quant_algo in (QuantAlgo.INT8, QuantAlgo.FP8):
        for _, parent in model.named_modules():
            if not is_attention_module(parent):
                continue

            assert isinstance(
                k_scale := getattr(parent, KVCacheScaleType.KEY.value), torch.nn.Parameter
            ) and isinstance(
                v_scale := getattr(parent, KVCacheScaleType.VALUE.value), torch.nn.Parameter
            ), "KV cache scales are not found in the attention module"
            # Note: We use literal string comparison for module name since compressed-tensors's is_attention_module()
            # function checks for these specific key names ("k_proj", "v_proj", "qkv_proj") to identify modules
            for name, child in parent.named_children():
                key = name.split(".")[-1]
                if key == "k_proj":
                    child.register_buffer("output_scale", k_scale.data)
                elif key == "v_proj":
                    child.register_buffer("output_scale", v_scale.data)
                elif key == "qkv_proj":
                    child.register_buffer("output_scale", torch.max(k_scale.data, v_scale.data))


def preprocess_qlinear_module(model: PreTrainedModel | PeftModel, quant_method: QuantizeMethod) -> None:
    """Unpacking the packed weight and zeros of the QLinear modules.

    Args:
        model (torch.nn.Module): The model to preprocess
        quant_method (QuantizeMethod): The quantization method used
    """
    replace_modules: dict[str, QuantLinear] = {}
    for name, module in model.named_modules():
        if isinstance(module, qlinear_cuda_old.QuantLinear | WQLinear_GEMM):
            bits = module.bits if isinstance(module, qlinear_cuda_old.QuantLinear) else module.w_bit
            module.register_buffer("unpacked_weight", unpack_qweight(module.qweight, bits, quant_method))
            module.register_buffer(
                "unpacked_zeros",
                unpack_zeros(module.qzeros, bits, quant_method) if isinstance(module.qzeros, torch.Tensor) else None,
            )
            if module.bias is not None and module.bias.dtype != model.dtype:
                module.bias = module.bias.to(model.dtype)
            if isinstance(module.scales, torch.Tensor) and module.scales.dtype != model.dtype:
                module.scales = module.scales.to(model.dtype)
            del module.qweight
            del module.qzeros

        elif isinstance(module, CompressedLinear):
            if "weight_packed" in module.compressor.compression_param_names:  # packed_compressor
                assert isinstance(packed_weight := module.weight_packed, torch.Tensor)
                assert isinstance(weight_shape := module.weight_shape, torch.Tensor)
                bits = 32 // (weight_shape[1].item() // packed_weight.shape[1])
                unpacked_weight = unpack_qweight(packed_weight, bits, quant_method).T.contiguous()
                unpacked_zeros = (
                    unpack_zeros(packed_zero, bits, quant_method)
                    if hasattr(module, "weight_zero_point")
                    and isinstance(packed_zero := module.weight_zero_point, torch.Tensor)
                    else None
                )
                del module.weight_packed
                if unpacked_zeros is not None:
                    del module.weight_zero_point
            else:
                unpacked_weight = module.weight.T.contiguous()
                unpacked_zeros = (
                    module.weight_zero_point
                    if hasattr(module, "weight_zero_point") and isinstance(module.weight_zero_point, torch.Tensor)
                    else None
                )
                del module.weight

            module.register_buffer("unpacked_weight", unpacked_weight)
            module.register_buffer("unpacked_zeros", unpacked_zeros)
            assert isinstance(weight_scale := module.weight_scale, torch.Tensor), "scale tensor is not found"
            module.register_buffer(
                "scales", weight_scale.T.contiguous() if module.quantization_scheme.weights.group_size else weight_scale
            )
            del module.weight_scale

        elif isinstance(module, _QuantLinear):
            replace_modules[name] = create_fake_quant_linear(module)

    def set_module(model: torch.nn.Module | torch.nn.ModuleList, module_path: str, new_module: torch.nn.Module) -> None:
        parts = module_path.split(".")
        for part in parts[:-1]:
            if part.isdigit():
                if isinstance(model, torch.nn.ModuleList):
                    model = model[int(part)]
                else:
                    raise TypeError(f"Module at '{'.'.join(parts[: parts.index(part)])}' is not indexable.")
            else:
                if not hasattr(model, part):
                    raise AttributeError(f"Module '{model.__class__.__name__}' does not have attribute '{part}'")
                model = getattr(model, part)
        setattr(model, parts[-1], new_module)

    for name, replace_module in replace_modules.items():
        set_module(model, name, replace_module)


def create_fake_quant_linear(module: _QuantLinear) -> QuantLinear:
    """Create a fake QuantLinear module.

    Args:
        module (_QuantLinear): The QuantLinear module of ModelOpt to create a new one from

    Returns:
        QuantLinear: The fake QuantLinear module
    """
    new_quant_linear = QuantLinear()
    if module.weight_quantizer.is_enabled:
        if (
            module.weight_quantizer._enable_pre_quant_scale
            and module.input_quantizer._enable_pre_quant_scale
            and hasattr(module.input_quantizer, "_pre_quant_scale")
        ):
            disable_pre_quant_scale_and_resmooth(module, True)
        weight, scale = export_modelopt_weight_and_scale(module)

        new_quant_linear.enable_weight_quantizer(
            weight=weight,
            num_bits=8 if module.weight_quantizer.num_bits == (4, 3) else module.weight_quantizer.num_bits,
            dynamic=module.weight_quantizer._dynamic,
            block_size=module.weight_quantizer.block_sizes.get(-1, None)
            if module.weight_quantizer.block_sizes is not None
            else None,
            scale=scale,
            zero_point=None,
        )
    else:
        new_quant_linear.weight = module.weight.data if isinstance(module.weight, torch.nn.Parameter) else module.weight

    if module.input_quantizer.is_enabled:
        if not module.input_quantizer._dynamic:
            assert (
                isinstance(amax := module.input_quantizer.amax, torch.Tensor) and amax.numel() == 1
            ), "Only per-tensor quantization for activation is supported"
            maxbound = torch.tensor([module.input_quantizer.maxbound])
            input_scale = amax.float() / maxbound.to(amax.device)
        else:
            input_scale = None

        new_quant_linear.enable_input_quantizer(
            8 if module.input_quantizer.num_bits == (4, 3) else module.input_quantizer.num_bits,
            module.input_quantizer._dynamic,
            scale=input_scale,
        )

    if module.output_quantizer.is_enabled:
        assert module.output_quantizer._dynamic is False, "Dynamic output quantization is not supported"
        assert (
            isinstance(amax := module.output_quantizer.amax, torch.Tensor) and amax.numel() == 1
        ), "Only per-tensor quantization for output is supported"
        maxbound = torch.tensor([module.output_quantizer.maxbound])
        output_scale = amax.float() / maxbound.to(amax.device)

        new_quant_linear.enable_output_quantizer(
            8 if module.output_quantizer.num_bits == (4, 3) else module.output_quantizer.num_bits,
            module.output_quantizer._dynamic,
            scale=output_scale,
        )

    new_quant_linear.bias = module.bias
    return new_quant_linear


def export_modelopt_weight_and_scale(module: _QuantLinear) -> tuple[torch.Tensor, torch.Tensor]:
    """Export the weight and scale of the QuantLinear module for ModelOpt.

    Args:
        module (_QuantLinear): The QuantLinear module to export

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The unpacked weight and scale
    """
    assert (
        hasattr(module, "weight")
        and isinstance(module.weight, torch.nn.Parameter)
        and isinstance(weight := module.weight.data, torch.Tensor)
        and isinstance(amax := module.weight_quantizer.amax, torch.Tensor)
    )
    if module.weight_quantizer.num_bits in (4, 8):
        maxbound = torch.tensor([module.weight_quantizer.maxbound])
        scale = (amax.float() / maxbound.to(amax.device)).to(amax.dtype)

        if module.weight_quantizer.block_sizes:
            assert (block_size := module.weight_quantizer.block_sizes.get(-1, None)) is not None
            scale = scale.reshape(-1, weight.shape[-1] // block_size)
            weight = (
                (weight / scale[..., :, torch.arange(weight.shape[-1]) // block_size])
                .round()
                .clamp(-maxbound.item() - 1, maxbound.item())
                .to(torch.int8)
            )
            scale = scale.T
        else:
            weight = (weight / scale).round().clamp(-maxbound.item() - 1, maxbound.item()).to(torch.int8)
            if scale.ndim == 1 and scale.numel() == 1:  # convert per-tensor to per-channel
                logger.warning(
                    "Converting per-tensor scale to per-channel scale, which might cause performance degradation."
                )
                scale = scale.expand(weight.shape[0], 1)

    else:
        maxbound = torch.tensor([module.weight_quantizer.maxbound])
        scale = (amax.float() / maxbound.to(amax.device)).to(amax.dtype)
        weight = (weight / scale).to(torch.float8_e4m3fn)

    if module.weight_quantizer.num_bits == 4:
        weight[weight < 0] += 16
        weight = weight.view(torch.uint8)

    return weight.T.contiguous(), scale.contiguous()


def unpack_qweight(qweight: torch.Tensor, bits: int, quant_method: QuantizeMethod) -> torch.Tensor:
    """Unpack the quantized weight tensor from int32 to int8 or uint8.

    if the weight is already unpacked, it will return the original tensor.

    Args:
        qweight (torch.Tensor): The quantized weight tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizeMethod): The quantization method used

    Returns:
        torch.Tensor: The unpacked weight tensor
    """
    assert bits in (4, 8), f"Unsupported bits: {bits=}"
    device = qweight.device
    if quant_method in ("gptq", "awq"):
        if bits == 4:
            qweight = unpack_int32_into_int8(
                qweight if quant_method == "awq" else qweight.T,
                quant_method == "awq",
            )
            qweight = qweight.T if quant_method == "gptq" else qweight
            qweight = qweight - 8
            qweight[qweight < 0] += 16
            qweight = qweight.view(torch.uint8).contiguous()
        else:
            qweight = (
                qweight.T.contiguous().view(torch.uint8).T.contiguous()
                if quant_method == "gptq"
                else qweight.view(torch.uint8).contiguous()
            )
            qweight = (qweight - 128).to(torch.int8)
    elif quant_method == "compressed-tensors":
        if qweight.dtype == torch.int32:
            original_shape = torch.Size([qweight.shape[0], qweight.shape[1] * (32 // bits)])
            qweight = unpack_from_int32(qweight, bits, original_shape)

            if bits == 4:
                qweight[qweight < 0] += 16
                qweight = qweight.view(torch.uint8).contiguous()
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    assert qweight.dtype.itemsize == 1, f"Wrong dtype of qweight: {qweight.dtype=}"
    return qweight.to(device)


def unpack_zeros(zeros: torch.Tensor, bits: int, quant_method: QuantizeMethod) -> torch.Tensor:
    """Unpack the quantized zero point tensor.

    Args:
        zeros (torch.Tensor): The quantized zero point tensor
        bits (int): The number of bits used for quantization
        quant_method (QuantizeMethod): The quantization method used

    Returns:
        torch.Tensor: The unpacked zero point tensor
    """
    device = zeros.device
    if quant_method in ("gptq", "awq", "compressed-tensors"):
        assert bits in (4, 8), f"Unsupported GPTQ or AWQ bits: {bits=}"
        if bits == 4:
            zeros = unpack_int32_into_int8(
                zeros.T if quant_method == "compressed-tensors" else zeros,
                quant_method == "awq",
            )
        else:
            zeros = zeros.view(torch.int8)
        zeros = -zeros + 2 ** (bits - 1) - 1 * (quant_method == "gptq")
    else:
        raise NotImplementedError(f"Unsupported quantization method: {quant_method}")

    assert zeros.dtype.itemsize == 1, f"Wrong dtype of zeros: {zeros.dtype=}"
    return zeros.to(device)


def is_quantized_by_modelopt(model: PreTrainedModel | PeftModel) -> bool:
    """Check if the model is quantized by ModelOpt.

    Args:
        model (PreTrainedModel | PeftModel): The model to check

    Returns:
        bool: True if the model is quantized by ModelOpt, False otherwise
    """
    for _, module in model.named_modules():
        if isinstance(module, _QuantLinear):
            return True
    return False


def get_modelopt_quantization_scheme(
    model: PreTrainedModel | PeftModel,
) -> tuple[QuantScheme | None, QuantScheme | None, QuantScheme | None]:
    """Get the quantization schemes for the model quantized by ModelOpt.

    Args:
        model (PreTrainedModel | PeftModel): The model to get the quantization schemes for

    Returns:
        tuple[QuantScheme | None, QuantScheme | None, QuantScheme | None]:
            The quantization schemes of input, weight, and output for the model
    """
    quant_schemes: list[tuple[QuantScheme | None, QuantScheme | None]] = []
    output_quant_schemes: list[QuantScheme] = []
    for _, module in model.named_modules():
        if not (
            isinstance(module, _QuantLinear)
            and (
                module.input_quantizer.is_enabled
                or module.weight_quantizer.is_enabled
                or module.output_quantizer.is_enabled
            )
        ):
            continue

        input_quant_scheme: QuantScheme | None = None
        weight_quant_scheme: QuantScheme | None = None
        if module.input_quantizer.is_enabled:
            assert (bits := module.input_quantizer.num_bits) == (
                4,
                3,
            ), f"Unsupported bits for activation quantization: {bits=}"
            assert (
                module.input_quantizer.axis is None or module.input_quantizer._dynamic
            ), "Only per-tensor or per-token quantization is supported for activation quantization"
            input_quant_scheme = QuantScheme(
                bits=8,
                mode=(QuantizeMode.PER_TOKEN if module.input_quantizer._dynamic else QuantizeMode.PER_TENSOR),
                type=QuantizeType.FLOAT,
                has_zero_point=False,
                dynamic=module.input_quantizer._dynamic,
            )

        if module.weight_quantizer.is_enabled:
            assert (bits := module.weight_quantizer.num_bits) in (
                4,
                8,
                (4, 3),
            ), f"Unsupported bits for weight quantization: {bits=}"
            weight_quant_scheme = QuantScheme(
                bits=8 if bits == (4, 3) else bits,
                mode=(
                    QuantizeMode.PER_GROUP
                    if module.weight_quantizer.block_sizes
                    else QuantizeMode.PER_TENSOR
                    if module.weight_quantizer.axis is None
                    else QuantizeMode.PER_CHANNEL
                ),
                type=QuantizeType.FLOAT if bits == (4, 3) else QuantizeType.INT,
                group_size=module.weight_quantizer.block_sizes.get(-1, None)
                if module.weight_quantizer.block_sizes
                else None,
                has_zero_point=False,
                dynamic=module.weight_quantizer._dynamic,
            )

        if module.output_quantizer.is_enabled:
            assert (bits := module.output_quantizer.num_bits) in (
                8,
                (4, 3),
            ), f"Unsupported bits for weight quantization: {bits=}"
            assert (
                module.output_quantizer.axis is None and not module.output_quantizer._dynamic
            ), "Only per-tensor quantization is supported for output quantization"
            output_quant_schemes.append(
                QuantScheme(
                    bits=8 if bits == (4, 3) else bits,
                    mode=QuantizeMode.PER_TENSOR,
                    type=QuantizeType.FLOAT if bits == (4, 3) else QuantizeType.INT,
                    has_zero_point=False,
                    dynamic=module.output_quantizer._dynamic,
                )
            )

        quant_schemes.append((input_quant_scheme, weight_quant_scheme))

    assert len(quant_schemes) < 2 or all(
        quant_schemes[0][0] == quant_scheme[0] and quant_schemes[0][1] == quant_scheme[1]
        for quant_scheme in quant_schemes[1:]
    ), "All quantization schemes must be the same"
    assert len(output_quant_schemes) < 2 or all(
        output_quant_schemes[0] == output_quant_scheme for output_quant_scheme in output_quant_schemes[1:]
    ), "All output quantization schemes must be the same"

    return (
        quant_schemes[0][0] if quant_schemes else None,
        quant_schemes[0][1] if quant_schemes else None,
        output_quant_schemes[0] if output_quant_schemes else None,
    )


def get_compressed_tensors_quantization_scheme(
    compressed_tensors_config: CompressedTensorsConfig,
) -> tuple[QuantScheme | None, QuantScheme | None, QuantScheme | None]:
    """Get the quantization schemes for the model quantized by CompressedTensors.

    Args:
        compressed_tensors_config (CompressedTensorsConfig): The compressed-tensors config

    Returns:
        tuple[QuantScheme | None, QuantScheme | None, QuantScheme | None]:
            The quantization schemes of input, weight, and output for the model
    """
    assert (
        compressed_tensors_config.quantization_config is not None
    ), "Quantization config for compressed-tensors is required"
    assert (
        len(compressed_tensors_config.quantization_config.config_groups) == 1
        and "group_0" in compressed_tensors_config.quantization_config.config_groups
        and isinstance(
            config := compressed_tensors_config.quantization_config.config_groups["group_0"],
            QuantizationScheme,
        )
    ), "More than one group is not supported yet"
    assert config.targets == [
        "Linear"
    ], f'Unsupported targets: {config.targets=}. Currently, only a single "Linear" target is supported.'

    input_quant_scheme: QuantScheme | None = None
    weight_quant_scheme: QuantScheme | None = None
    output_quant_scheme: QuantScheme | None = None

    if config.input_activations is not None:
        assert (
            config.input_activations.strategy is not None
            and config.input_activations.num_bits in (8, 16)
            and config.input_activations.strategy in (QuantizationStrategy.TENSOR, QuantizationStrategy.TOKEN)
        ), f"Unsupported input quantization scheme: {config.input_activations}"
        input_quant_scheme = QuantScheme(
            bits=config.input_activations.num_bits,
            mode=QuantizeMode.from_quant_strategy(config.input_activations.strategy),
            type=QuantizeType.from_quant_type(config.input_activations.type),
            dynamic=config.input_activations.dynamic,
        )

    if config.weights is not None:
        assert (
            config.weights.strategy is not None
            and config.weights.num_bits in (4, 8)
            and config.weights.strategy
            in (QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL, QuantizationStrategy.GROUP)
        ), f"Unsupported weight quantization scheme: {config.weights}"
        weight_quant_scheme = QuantScheme(
            bits=config.weights.num_bits,
            mode=QuantizeMode.from_quant_strategy(config.weights.strategy),
            type=QuantizeType.from_quant_type(config.weights.type),
            group_size=config.weights.group_size,
            has_zero_point=not config.weights.symmetric,
            dynamic=config.weights.dynamic,
        )

    if (kv_cache_config := compressed_tensors_config.quantization_config.kv_cache_scheme) is not None:
        assert (
            kv_cache_config.strategy is not None
            and kv_cache_config.num_bits == 8
            and kv_cache_config.strategy == QuantizationStrategy.TENSOR
            and kv_cache_config.symmetric is True
            and kv_cache_config.dynamic is False
        )
        output_quant_scheme = QuantScheme(
            bits=kv_cache_config.num_bits,
            mode=QuantizeMode.PER_TENSOR,
            type=QuantizeType.from_quant_type(kv_cache_config.type),
        )

    return input_quant_scheme, weight_quant_scheme, output_quant_scheme
