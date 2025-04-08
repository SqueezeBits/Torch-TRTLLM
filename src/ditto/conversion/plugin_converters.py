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

# mypy: disallow-untyped-decorators=False

from collections.abc import Sequence

import numpy as np
import tensorrt as trt
from tensorrt_llm.plugin import TRT_LLM_PLUGIN_NAMESPACE
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

from ..debug import enable_plugin_debug_info_hook
from ..fx.targets import (
    AllGatherPlugin,
    AllReducePlugin,
    Fp8RowwiseGemmPlugin,
    GemmPlugin,
    GPTAttentionPlugin,
    LoraPlugin,
    MixtureOfExpertsPlugin,
    Plugin,
    QuantizePerTokenPlugin,
    RecvPlugin,
    RmsnormQuantizationPlugin,
    SendPlugin,
    TopkLastDimPlugin,
    WeightOnlyGroupwiseQuantMatmulPlugin,
    WeightOnlyQuantMatmulPlugin,
)


@dynamo_tensorrt_converter(
    AllGatherPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_allgather_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert an AllGatherPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The AllGatherPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, AllGatherPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="allgather")


@dynamo_tensorrt_converter(
    AllReducePlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_allreduce_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert an AllReducePlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The AllReducePlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, AllReducePlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="allreduce")


@dynamo_tensorrt_converter(
    Fp8RowwiseGemmPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_fp8_rowwise_gemm_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a Fp8RowwiseGemmPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The Fp8RowwiseGemmPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, Fp8RowwiseGemmPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="fp8_rowwise_gemm")


@dynamo_tensorrt_converter(
    GemmPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_gemm_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a GemmPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The GemmPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, GemmPlugin)
    return _convert_plugin(
        ctx,
        target,
        args,
        kwargs,
        name,
        plugin_name="gemm",
    )


@dynamo_tensorrt_converter(
    GPTAttentionPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_gpt_attention_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a GPTAttentionPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The GPTAttentionPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, GPTAttentionPlugin)
    return _convert_plugin(
        ctx,
        target,
        args,
        kwargs,
        name,
        plugin_name="causal_attn",
    )


@dynamo_tensorrt_converter(
    LoraPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_lora_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a LoraPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The LoraPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, LoraPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="lora")


@dynamo_tensorrt_converter(
    QuantizePerTokenPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_quantize_per_token_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a QuantizePerTokenPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The QuantizePerTokenPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, QuantizePerTokenPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="quantize_per_token")


@dynamo_tensorrt_converter(
    RecvPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_recv_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a RecvPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The RecvPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, RecvPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="recv")


@dynamo_tensorrt_converter(
    RmsnormQuantizationPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_rmsnorm_quantization_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a RmsnormQuantizationPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The RmsnormQuantizationPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, RmsnormQuantizationPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="rmsnorm_quantized")


@dynamo_tensorrt_converter(
    SendPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_send_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a SendPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The SendPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, SendPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="send")


@dynamo_tensorrt_converter(
    WeightOnlyGroupwiseQuantMatmulPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_weight_only_groupwise_quant_matmul_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a WeightOnlyGroupwiseQuantMatmulPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The WeightOnlyGroupwiseQuantMatmulPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, WeightOnlyGroupwiseQuantMatmulPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="weight_only_groupwise_quant_matmul")


@dynamo_tensorrt_converter(
    WeightOnlyQuantMatmulPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_weight_only_quant_matmul_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a WeightOnlyQuantMatmulPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The WeightOnlyQuantMatmulPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, WeightOnlyQuantMatmulPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="weight_only_quant_matmul")


@dynamo_tensorrt_converter(
    MixtureOfExpertsPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_mixture_of_experts_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a MixtureOfExpertsPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The MixtureOfExpertsPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, MixtureOfExpertsPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="mixture_of_experts")


@dynamo_tensorrt_converter(
    TopkLastDimPlugin,
    supports_dynamic_shapes=True,
)
@enable_plugin_debug_info_hook
def convert_topk_last_dim_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a TopkLastDimPlugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The TopkLastDimPlugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    assert isinstance(target, TopkLastDimPlugin)
    return _convert_plugin(ctx, target, args, kwargs, name, plugin_name="topk_last_dim")


def _convert_plugin(
    ctx: ConversionContext,
    target: Plugin,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
    *,
    plugin_name: str,
    plugin_version: str = "1",
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a Plugin target to a TensorRT plugin layer.

    Args:
        ctx (ConversionContext): The conversion context
        target (Plugin): The Plugin target to convert
        args (tuple[Argument, ...]): Positional arguments to the plugin
        kwargs (dict[str, Argument]): Keyword arguments to the plugin
        name (str): Name for the plugin layer
        plugin_name (str): Name of the TensorRT plugin
        plugin_version (str, optional): Version of the plugin. Defaults to "1".

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Output tensor(s) from the plugin layer
    """
    creator_name = type(target).__name__.removesuffix("Plugin")  # type: ignore
    plugin_creator = trt.get_plugin_registry().get_plugin_creator(
        creator_name, plugin_version, TRT_LLM_PLUGIN_NAMESPACE
    )
    plugin_fields = target.get_fields()
    pfc = trt.PluginFieldCollection(plugin_fields)
    plugin = plugin_creator.create_plugin(plugin_name, pfc)
    assert plugin is not None
    args_kwargs = {f"arg_{i}": arg for i, arg in enumerate(args)}
    args_kwargs.update(kwargs)
    plugin_inputs: list[trt.ITensor] = [
        x if isinstance(x, trt.ITensor) else get_trt_tensor(ctx, x, f"{name}_{key}")
        for key, x in args_kwargs.items()
        if isinstance(x, trt.ITensor | np.ndarray)
    ]

    layer: trt.IPluginV2Layer = ctx.net.add_plugin_v2(plugin_inputs, plugin)
    layer.name = name
    if layer.num_outputs == 1:
        return layer.get_output(0)
    return [layer.get_output(i) for i in range(layer.num_outputs)]
