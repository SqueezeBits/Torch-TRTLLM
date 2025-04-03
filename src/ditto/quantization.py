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
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy
from loguru import logger
from tensorrt_llm.quantization import QuantAlgo
from torch._ops import OpOverload
from transformers import PreTrainedModel
from transformers.utils.quantization_config import (
    AwqConfig,
    CompressedTensorsConfig,
    GPTQConfig,
    QuantizationConfigMixin,
    QuantizationMethod,
)
from typing_extensions import Self

from .types import StrictlyTyped
from .modelopt import get_modelopt_modules, infer_modelopt_quanitzation

from peft import PeftModel

from .literals import ModelOptQuantFormat



class QuantizeMode(Enum):
    """Quantization mode.

    Attributes:
        PER_TENSOR (auto): Quantization mode for the quantized tensor.
        PER_GROUP (auto): Quantization mode for the quantized tensor.
        PER_CHANNEL (auto): Quantization mode for the quantized tensor.
        PER_TOKEN (auto): Quantization mode for the quantized tensor.
        UNKNOWN (auto): Quantization mode for the quantized tensor.
    """

    PER_TENSOR = auto()
    PER_GROUP = auto()
    PER_CHANNEL = auto()
    PER_TOKEN = auto()
    UNKNOWN = auto()


class QuantizeAlgorithm(Enum):
    """Quantization algorithm.

    Attributes:
        PTQ (auto): Post-training quantization.
        GPTQ (auto): GPTQ quantization.
        AWQ (auto): AWQ quantization.
        SMOOTHQUANT (auto): SmoothQuant quantization.
    """

    PTQ = auto()
    GPTQ = auto()
    AWQ = auto()
    SMOOTHQUANT = auto()

    ###### modelopt algorithms
    MODELOPT_QUANTIZATION_NVFP4 = auto()
    MODELOPT_QUANTIZATION_NVFP4_AWQ = auto()
    MODELOPT_QUANTIZATION_FP8 = auto()
    MODELOPT_QUANTIZATION_INT4_AWQ = auto()
    MODELOPT_QUANTIZATION_INT8_SQ = auto()
    MODELOPT_QUANTIZATION_W4A8_AWQ = auto()
    MODELOPT_QUANTIZATION_NONE = auto()

    @classmethod
    def from_hf_quant_method(cls, hf_quant_method: QuantizationMethod) -> "QuantizeAlgorithm":
        """Convert Hugging Face quantization method to Ditto quantization algorithm.

        Args:
            hf_quant_method (QuantizationMethod): The Hugging Face quantization method

        Returns:
            Self: The Ditto quantization algorithm
        """
        if hf_quant_method == QuantizationMethod.GPTQ:
            return cls.GPTQ
        if hf_quant_method == QuantizationMethod.AWQ:
            return cls.AWQ
        if hf_quant_method == QuantizationMethod.COMPRESSED_TENSORS:
            # Note: only support PTQ for compressed tensors currently
            return cls.PTQ

        raise NotImplementedError(f"Unsupported quantization method: {hf_quant_method}")


    def is_modelopt_algorithm(self) -> bool:
        """Check if current quantization is done by modelopt.

        Returns:
            bool: True if quantization is done by modelopt, False elsewise
        """
        return self.value >= QuantizeAlgorithm.MODELOPT_QUANTIZATION_NVFP4.value



class QuantScheme(StrictlyTyped):
    """Quantization scheme.

    Attributes:
        bits (int): The number of bits used for quantization.
        mode (QuantizeMode): The quantization mode.
        type (str | None): The type of the quantization.
        group_size (int | None): The size of the quantization group.
        has_zero_point (bool): Whether the quantization uses a zero point.
        dynamic (bool): Whether the quantization is dynamic.
    """

    bits: int
    mode: QuantizeMode
    type: str | None = None
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
        hf_quant_method (QuantizationMethod | None): The quantization method used, `None` means modelopt
        trtllm_quant_algo (QuantAlgo): The quantization algorithm used by TRT-LLM
        trtllm_kv_cache_quant_algo (QuantAlgo | None): The quantization algorithm used by TRT-LLM for the KV cache.
            Defaults to None.
        quant_configs (list[TargetQuantConfig]): The quantization schemes for the target operators.
    """

    hf_quant_method: QuantizationMethod | None
    modelopt_quant_format: ModelOptQuantFormat | None
    trtllm_quant_algo: QuantAlgo
    trtllm_kv_cache_quant_algo: QuantAlgo | None = None
    quant_configs: list[TargetQuantConfig] = []

    @classmethod
    def create_from(cls, model: PreTrainedModel | PeftModel) -> Self | None:
        """Create a GlobalQuantConfig from a pretrained config.

        Args:
            model (PreTrainedModel | PeftModel): The pretrained model

        Returns:
            Self | None: The created GlobalQuantConfig or None if no quantization config is found
        """

        modelopt_quant_modules = get_modelopt_modules(model)
        if modelopt_quant_modules:
            quant_format, quant_algo, group_size = infer_modelopt_quanitzation(modelopt_quant_modules)
            return cls(
                hf_quant_method=None,
                modelopt_quant_format=quant_format,
                trtllm_quant_algo=quant_algo,
                trtllm_kv_cache_quant_algo=None,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=None,
                        weight_quant_scheme=QuantScheme(
                            bits=4,
                            mode=QuantizeMode.PER_GROUP,
                            group_size=group_size,
                        ),
                    ),
                ],        
            )

        pretrained_config = model.config
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
                hf_quant_method=quantization_config.quant_method,
                modelopt_quant_format=None,
                trtllm_quant_algo=QuantAlgo.W4A16_GPTQ if quantization_config.bits == 4 else QuantAlgo.W8A16_GPTQ,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=None,
                        weight_quant_scheme=QuantScheme(
                            bits=quantization_config.bits,
                            mode=QuantizeMode.PER_GROUP,
                            group_size=quantization_config.group_size,
                        ),
                    ),
                ],
            )

        if isinstance(quantization_config, AwqConfig):
            if quantization_config.bits not in (4, 8):
                raise ValueError(f"Unsupported AWQ bits: {quantization_config.bits=}")
            return cls(
                hf_quant_method=quantization_config.quant_method,
                modelopt_quant_format=None,
                trtllm_quant_algo=QuantAlgo.W4A16_AWQ if quantization_config.bits == 4 else QuantAlgo.W8A16,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=None,
                        weight_quant_scheme=QuantScheme(
                            bits=quantization_config.bits,
                            mode=QuantizeMode.PER_GROUP,
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
            assert quantization_config.quantization_config.format in (
                CompressionFormat.float_quantized.value,
                CompressionFormat.naive_quantized.value,
            ), f"Unsupported compressed tensors format: {quantization_config.quantization_config.format}"
            assert (
                config.input_activations is not None
                and config.input_activations.strategy is not None
                and config.input_activations.strategy == QuantizationStrategy.TENSOR
                and config.input_activations.num_bits == 8
                and config.weights is not None
                and config.weights.strategy is not None
                and config.weights.strategy == QuantizationStrategy.TENSOR
                and config.weights.num_bits == 8
            ), "Currently, only per-tensor 8-bit quantization is supported"
            return cls(
                hf_quant_method=quantization_config.quantization_config.quant_method,
                modelopt_quant_format=None,
                trtllm_quant_algo=QuantAlgo.FP8,
                quant_configs=[
                    TargetQuantConfig(
                        target=torch.ops.aten.mm.default,
                        input_quant_scheme=QuantScheme(
                            bits=config.input_activations.num_bits,
                            mode=QuantizeMode.UNKNOWN,
                            dynamic=config.input_activations.dynamic,
                            type=config.input_activations.type,
                        ),
                        weight_quant_scheme=QuantScheme(
                            bits=config.weights.num_bits,
                            mode=QuantizeMode.UNKNOWN,
                            dynamic=config.weights.dynamic,
                            type=config.weights.type,
                        ),
                    ),
                ],
            )

        raise RuntimeError(f"Unsupported quantization algorithm: {quantization_config}")


def inference_trtllm_quant_algo(
    bits: int, compute_dtype: torch.dtype, *, hf_quant_method: QuantizationMethod | None
) -> QuantAlgo:
    """Infer the quantization algorithm for TensorRT-LLM .

    Args:
        bits (int): The number of bits used for quantization
        compute_dtype (torch.dtype): The compute data type
        hf_quant_method (QuantizationMethod | None): The quantization method used, `None` means modelopt

    Returns:
        QuantAlgo: The quantization algorithm for TensorRT-LLM
    """
    assert bits in (4, 8), "Only 4-bit and 8-bit quantization is supported for TensorRT-LLM"
    quant_algo: str = f"W{bits}A{compute_dtype.itemsize * 8}"
    if hf_quant_method == QuantizationMethod.GPTQ:
        quant_algo = f"{quant_algo}_GPTQ"
    elif hf_quant_method == QuantizationMethod.AWQ:
        quant_algo = f"{quant_algo}_AWQ"
    elif hf_quant_method is None: # modelopt
        quant_algo = f"{quant_algo}_AWQ"
    else:
        raise RuntimeError(f"Unsupported quantization method: {hf_quant_method}")

    assert quant_algo in QuantAlgo, f"Unsupported quantization algorithm: {quant_algo}"
    return QuantAlgo[quant_algo]


def resolve_qlinear_device_map(model: torch.nn.Module) -> None:
    """Resolve the device map for the QuantLinear module.

    Args:
        model (torch.nn.Module): The model to resolve the device map for
    """
    # Note: This is a temporary solution to resolve PendingUnbackedSymbolNotFound error during the fake propagation.
    for _, module in model.named_modules():
        if isinstance(module, qlinear_cuda_old.QuantLinear):
            if module.wf.device != module.qzeros.device:
                module.wf = module.wf.to(module.qzeros.device)
