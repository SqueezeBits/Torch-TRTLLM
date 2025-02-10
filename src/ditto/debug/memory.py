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
from typing import Literal, overload

import psutil
import torch

from ..types import DeviceLikeType


@overload
def get_memory_footprint(*devices: DeviceLikeType) -> str:
    ...


@overload
def get_memory_footprint(*devices: DeviceLikeType, dumps: Literal[False] = False) -> dict[str, dict[str, float]]:
    ...


def get_memory_footprint(  # type: ignore[misc]
    *devices: DeviceLikeType,
    dumps: bool = True,
) -> str | dict[str, dict[str, int]]:
    """Get the memory footprint of the devices.

    Args:
        *devices (DeviceLikeType): The devices to get the memory footprint of.
        dumps (bool): Whether to dump the memory footprint as a JSON string.

    Returns:
        str | dict[str, dict[str, int]]: The memory footprint of the devices.
    """
    memory_footprint = {"cpu": get_host_memory_footprint()}
    for device in {torch.device(d) for d in devices}:
        memory_footprint.update({f"{device}": get_device_memory_footprint(device)})
    if dumps:
        return json.dumps(
            {device: {k: format_size(v) for k, v in entry.items()} for device, entry in memory_footprint.items()},
            indent=2,
            sort_keys=True,
        )
    return memory_footprint


def get_host_memory_footprint() -> dict[str, int]:
    """Get the memory footprint of the host.

    Returns:
        dict[str, int]: The memory footprint of the host.
    """
    # Get the current process ID
    pid = os.getpid()
    # Create a Process object for the current process
    process = psutil.Process(pid)
    # Get memory usage details
    return process.memory_info()._asdict()


def get_device_memory_footprint(device: DeviceLikeType) -> dict[str, int]:
    """Get the memory footprint of the device.

    Args:
        device (DeviceLikeType): The device to get the memory footprint of.

    Returns:
        dict[str, int]: The memory footprint of the device.
    """
    return {
        "Allocated Memory": torch.cuda.memory_allocated(device),
        "Max Allocated Memory": torch.cuda.max_memory_allocated(device),
        "Reserved Memory": torch.cuda.memory_reserved(device),
    }


def format_size(size_in_bytes: float) -> str:
    """Format the size in bytes to a human-readable string.

    Args:
        size_in_bytes (float): The size in bytes to format.

    Returns:
        str: The formatted size.
    """
    if size_in_bytes < (1 << 20):
        return f"{size_in_bytes / (1 << 10):.3f} KB"
    if size_in_bytes < (1 << 30):
        return f"{size_in_bytes / (1 << 20):.3f} MB"
    return f"{size_in_bytes / (1 << 30):.3f} GB"
