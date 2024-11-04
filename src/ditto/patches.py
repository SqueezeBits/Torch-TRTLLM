import json
from typing import Any

import tensorrt as trt
import tensorrt_llm as trtllm

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
    messages.append(get_network_ir(self._trt_network))
    network_ir = "\n".join(messages)
    if not path:
        return network_ir
    network_path = path.replace(".dot", ".txt")
    if not network_path.endswith(".txt"):
        network_path = f"{network_path}.txt"
    with open(network_path, "w") as f:
        f.write(network_ir)
    trtllm.logger.info(f"Network IR saved at {network_path}")
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


trtllm.Network.to_dot = patched_trtllm_network_to_dot

trtllm.Builder.build_engine = patched_builder_build_engine
