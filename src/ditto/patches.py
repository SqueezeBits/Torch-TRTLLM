import json
from typing import Any

import numpy as np
import onnx
import tensorrt as trt
import tensorrt_llm as trtllm
from tensorrt_llm.functional import RopeEmbeddingUtils, RotaryScalingType

from .pretty_print import builder_config_as_dict, get_network_ir


def patched_trtllm_network_to_dot(self: trtllm.Network, path: str | None) -> str | None:
    messages: list[str] = []
    if input_tensors := self._inputs:
        num_profiles = len(list(input_tensors.values())[0].profiles)
        for i in range(num_profiles):
            for input_name, input_tensor in input_tensors.items():
                if len(input_tensor.profiles) == 0:
                    continue
                shape_profile = input_tensor.profiles[i]
                messages.append(f"# Profile {i} for '{input_name}':")
                messages.append(f"#   Min shape: {(*shape_profile.min,)}")
                messages.append(f"#   Opt shape: {(*shape_profile.opt,)}")
                messages.append(f"#   Max shape: {(*shape_profile.max,)}")
    code, model_proto = get_network_ir(self._trt_network)
    messages.append(code)
    network_ir = "\n".join(messages)
    if not path:
        return network_ir
    name = self._trt_network.name or "unnamed_network"
    with open(f"{name}.txt", "w") as f:
        f.write(network_ir)
    trtllm.logger.info(f"Network IR saved at {name}.txt")
    with open(f"{name}.onnx", "wb") as f:
        onnx.save(model_proto, f, save_as_external_data=True, location=f"{name}.bin")
    trtllm.logger.info(f"Network ONNX saved at {name}.onnx")
    return None


original_builder_build_engine = trtllm.Builder.build_engine


def patched_builder_build_engine(
    self: trtllm.Builder,
    network: trtllm.Network,
    builder_config: trtllm.BuilderConfig,
    managed_weights: dict[str, Any] | None = None,
) -> trt.IHostMemory:
    engine = original_builder_build_engine(self, network, builder_config, managed_weights)
    with open(path := "builder_config.json", "w") as f:
        config_dict = builder_config_as_dict(builder_config.trt_builder_config)
        json.dump(config_dict, f, indent=2, sort_keys=True)
    trtllm.logger.info(f"trt.IBuilderConfig saved at {path}")
    return engine


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
    print("=========ROPE constant inputs=========")
    print(f"rotary_embedding_max_positions: {num_pos}")
    print(f"rotary_embedding_dim: {dim}")
    print(f"rotary_embedding_base: {theta}")
    print(f"rotary_embedding_scale: {scale}")
    print(f"rotary_embedding_scale_type: {scale_type.name}")
    print(f"llama3_scaling_config: {rope_scaling_config}")
    print("======================================")
    return original_rope_embedding_utils_create_sinusoidal_positions_for_attention_plugin(
        num_pos, dim, theta, scale, scale_type, rope_scaling_config, dtype
    )


trtllm.Network.to_dot = patched_trtllm_network_to_dot

trtllm.Builder.build_engine = patched_builder_build_engine

RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin = (
    patched_create_sinusoidal_positions_for_attention_plugin
)
