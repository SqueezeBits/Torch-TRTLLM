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

from typing import Any

import tensorrt as trt

BuiltinConst = bool | float | int | str | None


def builder_config_as_dict(builder_config: trt.IBuilderConfig) -> dict[str, Any]:
    """Save all attributes of a TensorRT IBuilderConfig object as a JSON file.

    Args:
        builder_config (trt.IBuilderConfig): The TensorRT builder configuration to save.
        file_path (str): The path to the JSON file where the configuration will be saved.
    """
    # Dictionary to store attribute values
    config_data: dict[str, Any] = {}

    def normalize(value: Any) -> BuiltinConst | list[Any] | tuple[Any, ...] | dict[str, Any]:
        value = getattr(value, "name", value)
        if isinstance(value, list):
            return [normalize(x) for x in value]
        if isinstance(value, tuple):
            return tuple(normalize(x) for x in value)
        if isinstance(value, dict):
            return {f"{k}": normalize(v) for k, v in value.items()}
        if not isinstance(value, BuiltinConst):
            return f"{value}"
        return value

    # Loop through attributes in IBuilderConfig and retrieve their values
    for attr in dir(builder_config):
        # Filter out private attributes, methods, and unsupported types
        if not attr.startswith("_") and not callable(getattr(builder_config, attr)):
            try:
                # Retrieve attribute value
                value = getattr(builder_config, attr)
                if attr == "flags" and isinstance(value, int):
                    value = {
                        f"{flag.value:02d}:{name}": builder_config.get_flag(flag)
                        for name, flag in trt.BuilderFlag.__members__.items()
                    }
                config_data[attr] = normalize(value)
            except Exception as e:
                # Handle any errors in retrieving attribute value
                config_data[attr] = f"Error retrieving: {str(e)}"

    return config_data


def get_human_readable_flags(network: trt.INetworkDefinition) -> dict[str, bool]:
    """Get the human-readable flags of a TensorRT network.

    Args:
        network (trt.INetworkDefinition): The TensorRT network to get the flags of.

    Returns:
        dict[str, bool]: The human-readable flags of the TensorRT network.
    """
    return {
        f"{flag.value:02d}:{name}": network.get_flag(flag)
        for name, flag in trt.NetworkDefinitionCreationFlag.__members__.items()
    }
