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

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
import tensorrt_llm as trtllm
import torch
from loguru import logger
from tensorrt_llm import default_net
from tensorrt_llm.functional import (
    RopeEmbeddingUtils,
    RotaryScalingType,
)
from tensorrt_llm.runtime.generation import GenerationSession

from ..debug import (
    builder_config_as_dict,
    open_debug_artifact,
    save_for_debug,
)


def patched_trtllm_network_to_dot(self: trtllm.Network, path: Path | str | None) -> str | None:
    if path is None:
        return None
    save_for_debug(f"trt_network_def_{Path(path).stem}", self.trt_network)
    return None


original_builder_build_engine = trtllm.Builder.build_engine


def patched_builder_build_engine(
    self: trtllm.Builder,
    network: trtllm.Network,
    builder_config: trtllm.BuilderConfig,
    managed_weights: dict[str, Any] | None = None,
) -> trt.IHostMemory:
    if layer_names_ := os.environ.get("TRTLLM_ADD_OUTPUT", None):
        try:
            layer_names: dict[str, str] = json.loads(layer_names_)
        except json.JSONDecodeError:
            logger.error(f"Invalid json provided to TRTLLM_ADD_OUTPUT: {layer_names_}")
            layer_names = {}
        if layer_names:
            net = network.trt_network
            for layer_idx in range(net.num_layers):
                layer = net.get_layer(layer_idx)
                if layer.name not in layer_names:
                    continue
                layer_alias = layer_names.pop(layer.name)
                for output_idx in range(layer.num_outputs):
                    output = layer.get_output(output_idx)
                    if layer.num_outputs > 1:
                        layer_alias = f"{layer_alias}_{output_idx}"
                    logger.info(f"Marking new output: {output.name} -> {layer_alias}")
                    net.mark_output(output)
                    output.name = layer_alias
            if layer_names:
                for layer_name in layer_names:
                    logger.error(f"No such layer found: {layer_name}")
                logger.info("The layer names are as follows:")
                print("\n".join(net.get_layer(layer_idx).name for layer_idx in range(net.num_layers)))
    serialized_engine = original_builder_build_engine(self, network, builder_config, managed_weights)
    with open_debug_artifact("builder_config.json") as f:
        if f:
            config_dict = builder_config_as_dict(builder_config.trt_builder_config)
            json.dump(config_dict, f, indent=2, sort_keys=True)
    with open_debug_artifact("trtllm_builder_config.json", "w") as f:
        if f:
            config_dict = builder_config.to_dict()
            if "quant_mode" in config_dict["builder_config"]:
                config_dict["builder_config"]["quant_mode"] = [
                    obj.to_dict() for obj in config_dict["builder_config"]["quant_mode"].objs
                ]
            json.dump(config_dict, f, indent=2, sort_keys=True)
    save_for_debug("trt_engine", serialized_engine)
    return serialized_engine


original_rope_embedding_utils_create_sinusoidal_positions_for_attention_plugin = (
    RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin
)


def patched_create_sinusoidal_positions_for_attention_plugin(
    num_pos: int,
    dim: int,
    theta: float = 10000.0,
    scale: float = 1.0,
    scale_type: RotaryScalingType = RotaryScalingType.none,
    rope_scaling_config: dict[str, Any] | None = None,
    dtype: type[np.number] = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    rotary_inv_freq, embed_positions = original_rope_embedding_utils_create_sinusoidal_positions_for_attention_plugin(
        num_pos, dim, theta, scale, scale_type, rope_scaling_config, dtype
    )
    with open_debug_artifact("rope_inputs.pt", "wb") as f:
        if f:
            torch.save(
                {
                    "rotary_inv_freq": torch.from_numpy(rotary_inv_freq),
                    "rotary_cos_sin": torch.from_numpy(embed_positions),
                },
                f,
            )
    return rotary_inv_freq, embed_positions


def patched_dump_debug_buffers(self: GenerationSession, step: int) -> None:
    debug_buffer = {**self.debug_buffer}
    if "host_kv_cache_pool_pointers" in debug_buffer:
        debug_buffer["kv_cache_pool"] = self.kv_cache_pool
    with open_debug_artifact(f"step{step}.pt", "wb") as f:
        if f:
            torch.save(debug_buffer, f)
    for name, value in debug_buffer.items():
        print(
            f"{name}: {value if (value.ndim < 3 or value.numel() < 100) else f'tensor with shape={(*value.shape,)}, dtype={value.dtype}'}"
        )


original_plugin_creator_create_plugin = trt.IPluginCreator.create_plugin


PLUGINS_TO_FIELDS: list[tuple[trt.IPluginV2, list[str]]] = []


def patched_plugin_creator_create_plugin(
    self: trt.IPluginCreator, plugin_name: str, pfc: trt.PluginFieldCollection
) -> trt.IPluginV2:
    plugin = original_plugin_creator_create_plugin(self, plugin_name, pfc)
    fields = [
        f"{field.name} ({field.type}): {field.data} (dtype={field.data.dtype}, shape={field.data.shape})"
        for field in pfc
    ]
    PLUGINS_TO_FIELDS.append((plugin, fields))
    return plugin


original_network_definition_add_plugin_v2 = trt.INetworkDefinition.add_plugin_v2


def patched_network_definition_add_plugin_v2(
    self: trt.INetworkDefinition, inputs: list[trt.ITensor], plugin: trt.IPluginV2
) -> trt.IPluginV2Layer:
    layer = original_network_definition_add_plugin_v2(self, inputs, plugin)
    default_net()._set_layer_name(layer)
    with open_debug_artifact(f"plugins/{layer.name}.json", "w") as f:
        if f:
            pfc_idx = None
            for i, (p, _) in enumerate(PLUGINS_TO_FIELDS):
                if plugin is p:
                    pfc_idx = i
                    break
            json.dump(
                {
                    "namespace": plugin.plugin_namespace,
                    "plugin_type": plugin.plugin_type,
                    "inputs": [f"ITensor(name={t.name}, dtype={t.dtype.name}, shape={t.shape})" for t in inputs],
                    "fields": PLUGINS_TO_FIELDS.pop(pfc_idx)[1] if pfc_idx is not None else [],
                },
                f,
                indent=2,
            )
    return layer


trtllm.Network.to_onnx = patched_trtllm_network_to_dot

trtllm.Builder.build_engine = patched_builder_build_engine

trt.IPluginCreator.create_plugin = patched_plugin_creator_create_plugin
trt.INetworkDefinition.add_plugin_v2 = patched_network_definition_add_plugin_v2


RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin = (
    patched_create_sinusoidal_positions_for_attention_plugin
)

GenerationSession.dump_debug_buffers = patched_dump_debug_buffers


logger.info("ditto patches for TensorRT-LLM are applied!")
