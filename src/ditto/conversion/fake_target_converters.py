# mypy: disallow-untyped-decorators=False

from collections.abc import Sequence

import numpy as np
import tensorrt as trt
from tensorrt_llm.functional import PluginInfo, set_plugin_info
from tensorrt_llm.plugin import TRT_LLM_PLUGIN_NAMESPACE
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,
    set_layer_name,
)

from ..debug import open_debug_artifact
from ..fx.targets import GPTAttentionPlugin


@dynamo_tensorrt_converter(
    GPTAttentionPlugin,
    supports_dynamic_shapes=True,
)
def convert_fake_gpt_attention_plugin(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    assert isinstance(target, GPTAttentionPlugin)
    plugin_creator = trt.get_plugin_registry().get_plugin_creator("GPTAttention", "1", TRT_LLM_PLUGIN_NAMESPACE)
    plugin_fields = target.get_plugin_fields()
    pfc = trt.PluginFieldCollection(plugin_fields)
    attn_plugin = plugin_creator.create_plugin("causal_attn", pfc)
    assert attn_plugin is not None
    args_kwargs = {f"arg_{i}": arg for i, arg in enumerate(args)}
    args_kwargs.update(kwargs)
    plugin_inputs: list[trt.ITensor] = [
        x if isinstance(x, trt.ITensor) else get_trt_tensor(ctx, x, f"{name}_{key}")
        for key, x in args_kwargs.items()
        if isinstance(x, trt.ITensor | np.ndarray)
    ]

    if target.layer_idx == 0:
        with open_debug_artifact("plugin.txt") as f:
            if f:
                f.writelines(
                    (
                        "plugin field collection:\n",
                        "\n".join(
                            f"{field.name} ({field.type}): {field.data} "
                            f"(dtype={field.data.dtype}, shape={field.data.shape})"
                            for field in pfc
                        ),
                        "\nplugin inputs:\n",
                        "\n".join(
                            f"ITensor(name={t.name}, dtype={t.dtype.name}, shape={t.shape})" for t in plugin_inputs
                        ),
                    )
                )

    layer = ctx.net.add_plugin_v2(plugin_inputs, attn_plugin)
    plugin_info = PluginInfo(plugin_creator, "causal_attn", pfc)
    set_plugin_info(ctx.net, layer.name, plugin_info)
    set_layer_name(layer, target, name, SourceIR.UNKNOWN)
    return layer.get_output(0)
