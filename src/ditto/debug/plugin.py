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

import contextlib
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from functools import cached_property, wraps
from typing import Any, TypeVar

import tensorrt as trt

from .io import open_debug_artifact, should_save_debug_artifacts


@contextlib.contextmanager
def plugin_debug_info_hook(name: str) -> Generator[None, None, None]:
    """Context manager that hooks into TensorRT plugin creation to save debug info.

    Args:
        name (str): Name to use for the plugin debug artifacts

    Yields:
        None: This context manager yields nothing

    Returns:
        Generator[None, None, None]: A generator that yields nothing
    """
    if not should_save_debug_artifacts():
        yield None
        return

    try:
        hold_plugin_fields = HoldPluginFields()
        save_plugin = SavePlugin(name, hold_plugin_fields)
        trt.IPluginCreator.create_plugin = hold_plugin_fields.patch
        trt.INetworkDefinition.add_plugin_v2 = save_plugin.patch
        yield None
    finally:
        trt.IPluginCreator.create_plugin = hold_plugin_fields.origin
        trt.INetworkDefinition.add_plugin_v2 = save_plugin.origin


F = TypeVar("F", bound=Callable)


def enable_plugin_debug_info_hook(func: F) -> F:
    """Enable plugin debug info hooks for a function.

    Args:
        func (FuncType): Function to decorate

    Returns:
        FuncType: Decorated function that enables plugin debug info hooks
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        name = args[4] if len(args) > 4 else kwargs.get("name")
        if name is None:
            name = f"unknown_plugin_{uuid.uuid4()}"
        with plugin_debug_info_hook(name):
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class Patch(ABC):
    """Abstract base class for patching TensorRT plugin functions.

    Args:
        origin (Callable[..., Any]): Original function to patch
    """

    def __init__(self, origin: Callable[..., Any]) -> None:
        self.origin = origin

    @cached_property
    @abstractmethod
    def patch(self) -> Callable[..., Any]:
        """Get the patched version of the function.

        Returns:
            Callable[..., Any]: Patched function
        """


class HoldPluginFields(Patch):
    """Patch that captures plugin fields during plugin creation.

    Args:
        None
    """

    def __init__(self) -> None:
        super().__init__(trt.IPluginCreator.create_plugin)
        self.plugins_to_fields: list[tuple[trt.IPluginV2, list[str]]] = []

    @cached_property
    def patch(self) -> Callable[..., trt.IPluginV2]:
        """Get patched version of plugin creator function that captures fields.

        Returns:
            Callable[..., trt.IPluginV2]: Patched plugin creator function
        """

        def patched_plugin_creator_create_plugin(
            creator: trt.IPluginCreator, plugin_name: str, pfc: trt.PluginFieldCollection
        ) -> trt.IPluginV2:
            plugin = self.origin(creator, plugin_name, pfc)
            fields = [
                f"{field.name} ({field.type}): {field.data} (dtype={field.data.dtype}, shape={field.data.shape})"
                for field in pfc
            ]
            self.plugins_to_fields.append((plugin, fields))
            return plugin

        return patched_plugin_creator_create_plugin


class SavePlugin(Patch):
    """Patch that saves plugin debug info to files.

    Args:
        name (str): Name to use for debug artifacts
        hold_plugin_fields (HoldPluginFields): Instance holding captured plugin fields
    """

    def __init__(self, name: str, hold_plugin_fields: HoldPluginFields) -> None:
        super().__init__(trt.INetworkDefinition.add_plugin_v2)
        self.hold_plugin_fields = hold_plugin_fields
        self.name = name

    @cached_property
    def patch(self) -> Callable[..., trt.IPluginV2Layer]:
        """Get patched version of add_plugin_v2 that saves debug info.

        Returns:
            Callable[..., trt.IPluginV2Layer]: Patched add_plugin_v2 function
        """

        def patched_network_definition_add_plugin_v2(
            net: trt.INetworkDefinition, inputs: list[trt.ITensor], plugin: trt.IPluginV2
        ) -> trt.IPluginV2Layer:
            layer = self.origin(net, inputs, plugin)
            layer.name = self.name
            plugins_to_fields = self.hold_plugin_fields.plugins_to_fields
            with open_debug_artifact(f"plugins/{layer.name}.json", "w") as f:
                if f:
                    pfc_idx = None
                    for i, (p, _) in enumerate(plugins_to_fields):
                        if plugin is p:
                            pfc_idx = i
                            break
                    json.dump(
                        {
                            "namespace": plugin.plugin_namespace,
                            "plugin_type": plugin.plugin_type,
                            "inputs": [
                                f"ITensor(name={t.name}, dtype={t.dtype.name}, shape={t.shape})" for t in inputs
                            ],
                            "fields": plugins_to_fields.pop(pfc_idx)[1] if pfc_idx is not None else [],
                        },
                        f,
                        indent=2,
                    )
            return layer

        return patched_network_definition_add_plugin_v2
