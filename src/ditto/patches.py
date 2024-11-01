import json
from typing import Any

import tensorrt as trt
import tensorrt_llm as trtllm

from .pretty_print import builder_config_as_dict, get_network_ir


def patched_trtllm_network_to_dot(self: trtllm.Network, path: str):
    network_path = path.replace(".dot", ".txt")
    network_ir = get_network_ir(self._trt_network)
    with open(network_path, "w") as f:
        f.write(network_ir)
    trtllm.logger.info(f"Network IR saved at {network_path}")


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
