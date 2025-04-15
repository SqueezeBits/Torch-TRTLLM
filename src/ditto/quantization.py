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
from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy, QuantizationType
from loguru import logger
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.functional import unpack_int32_into_int8
from torch._ops import OpOverload
from transformers import PretrainedConfig
from transformers.utils.quantization_config import (
    AwqConfig,
    CompressedTensorsConfig,
    GPTQConfig,
    QuantizationConfigMixin,
)
from typing_extensions import Self

from .literals import HFQuantizeMethod
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
        hf_quant_method (HFQuantizeMethod): The quantization method used by the Hugging Face model
        trtllm_quant_algo (QuantAlgo): The quantization algorithm used by TRT-LLM
        trtllm_kv_cache_quant_algo (QuantAlgo | None): The quantization algorithm used by TRT-LLM for the KV cache.
            Defaults to None.
        clamp_val (list[float] | None): The clamp values for the quantization. Defaults to None.
        quant_configs (list[TargetQuantConfig]): The quantization schemes for the target operators.
    """

    hf_quant_method: HFQuantizeMethod
    trtllm_quant_algo: QuantAlgo
    trtllm_kv_cache_quant_algo: QuantAlgo | None = None
    clamp_val: list[float] | None = None
    quant_configs: list[TargetQuantConfig] = []

    @classmethod
    # pylint: disable-next=too-many-branches
    def create_from(cls, pretrained_config: PretrainedConfig) -> Self | None:
        """Create a GlobalQuantConfig from a pretrained config.

        Args:
            pretrained_config (PretrainedConfig): The pretrained config

        Returns:
            Self | None: The created GlobalQuantConfig or None if no quantization config is found
        """
        if not (
            (quantization_config := getattr(pretrained_config, "quantization_config", None)) is not None
            and isinstance(quantization_config, QuantizationConfigMixin)
        ):
            return None

        logger.info("Quantization config is found in the pretrained config")
        if isinstance(quantization_config, GPTQConfig):
            if quantization_config.bits not in (4, 8):
                raise ValueError(f"Unsupported GPTQ bits: {quantization_config.bits=}")
            return cls(
                hf_quant_method=quantization_config.quant_method.value,
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
                hf_quant_method=quantization_config.quant_method.value,
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

        if (
            isinstance(quantization_config, CompressedTensorsConfig)
            and quantization_config.quantization_config is not None
            and len(quantization_config.quantization_config.config_groups) == 1
            and isinstance(
                config := quantization_config.quantization_config.config_groups["group_0"],
                QuantizationScheme,
            )
        ):
            assert config.targets == [
                "Linear"
            ], f'Unsupported targets: {config.targets=}. Currently, only a single "Linear" target is supported.'

            input_quant_scheme: QuantScheme | None = None
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
            weight_quant_scheme: QuantScheme | None = None
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

            assert weight_quant_scheme is not None, "Weight quantization scheme is required"

            if input_quant_scheme is None:
                if weight_quant_scheme.type == QuantizeType.INT:
                    trtllm_quant_algo = QuantAlgo.W8A16 if weight_quant_scheme.bits == 8 else QuantAlgo.W4A16
                else:
                    raise NotImplementedError(f"Unsupported weight-only quantization type: {weight_quant_scheme.type=}")
            else:
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
            return cls(
                hf_quant_method=quantization_config.quantization_config.quant_method,
                trtllm_quant_algo=trtllm_quant_algo,
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


def preprocess_qlinear_module(model: torch.nn.Module, global_quant_config: GlobalQuantConfig) -> None:
    """Unpacking the packed weight and zeros of the QLinear modules.

    Args:
        model (torch.nn.Module): The model to preprocess
        global_quant_config (GlobalQuantConfig): The global quantization config
    """
    for _, module in model.named_modules():
        if isinstance(module, qlinear_cuda_old.QuantLinear | WQLinear_GEMM):
            bits = module.bits if isinstance(module, qlinear_cuda_old.QuantLinear) else module.w_bit
            module.register_buffer(
                "unpacked_weight",
                unpack_qweight(
                    module.qweight,
                    bits,
                    global_quant_config.hf_quant_method,
                ),
            )
            module.register_buffer(
                "unpacked_zeros",
                unpack_zeros(
                    module.qzeros,
                    bits,
                    global_quant_config.hf_quant_method,
                ),
            )
            if (
                global_quant_config.quant_configs[0].weight_quant_scheme is not None
                and global_quant_config.quant_configs[0].weight_quant_scheme.mode == QuantizeMode.UNKNOWN
            ):
                global_quant_config.quant_configs[0].weight_quant_scheme.mode = QuantizeMode.PER_GROUP
        else:
            pass


def unpack_qweight(qweight: torch.Tensor, bits: int, quant_method: HFQuantizeMethod) -> torch.Tensor:
    """Unpack the quantized weight tensor from int32 to int8 or uint8.

    if the weight is already unpacked, it will return the original tensor.

    Args:
        qweight (torch.Tensor): The quantized weight tensor
        bits (int): The number of bits used for quantization
        quant_method (HFQuantizeMethod): The quantization method used

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


def unpack_zeros(zeros: torch.Tensor, bits: int, quant_method: HFQuantizeMethod) -> torch.Tensor:
    """Unpack the quantized zero point tensor.

    Args:
        zeros (torch.Tensor): The quantized zero point tensor
        bits (int): The number of bits used for quantization
        quant_method (HFQuantizeMethod): The quantization method used

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
