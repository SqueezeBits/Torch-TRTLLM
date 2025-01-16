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
from ..fx.targets import AllGatherPlugin, AllReducePlugin, GemmPlugin, GPTAttentionPlugin, Plugin


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
    creator_name = type(target).__name__.removesuffix("Plugin")
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

    layer = ctx.net.add_plugin_v2(plugin_inputs, plugin)
    layer.name = name
    return layer.get_output(0)
