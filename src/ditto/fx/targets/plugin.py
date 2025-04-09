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

import re
from abc import ABC, abstractmethod
from enum import IntEnum, IntFlag
from typing import Any

import numpy as np
import tensorrt as trt
import torch

from ...types import StrictlyTyped

PLUGIN_FIELD_TYPES: dict[type[np.number], trt.PluginFieldType] = {
    np.int8: trt.PluginFieldType.INT8,
    np.int16: trt.PluginFieldType.INT16,
    np.int32: trt.PluginFieldType.INT32,
    np.float16: trt.PluginFieldType.FLOAT16,
    np.float32: trt.PluginFieldType.FLOAT32,
    np.float64: trt.PluginFieldType.FLOAT64,
}


class Plugin(StrictlyTyped, ABC):
    """Base class for TensorRT plugins.

    Provides common functionality for converting Python objects to TensorRT plugin fields
    and handling plugin attributes.
    """

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for a plugin field value.

        Args:
            name (str): Name of the field
            value (Any): Value to get dtype for

        Returns:
            type[np.number]: numpy dtype for the value

        Raises:
            ValueError: If dtype cannot be inferred for the value type
        """
        if isinstance(value, bool | IntEnum):
            return np.int8
        if isinstance(value, trt.DataType | IntFlag | int):
            return np.int32
        if isinstance(value, float):
            return np.float32
        if isinstance(value, list | tuple):
            if not value:
                raise ValueError("Empty sequence for field {name} of {cls.__name__}")
            first_dtype = cls.get_field_dtype(name, value[0])
            if not all(cls.get_field_dtype(name, v) == first_dtype for v in value[1:]):
                raise ValueError(
                    f"All elements in sequence for field {name} of {cls.__name__} must have the same dtype"
                )
            return first_dtype
        raise ValueError(f"Cannot infer dtype for field {name} of {cls.__name__}: {type(value)}")

    @classmethod  # pylint: disable-next=unused-argument
    def process_value(cls, name: str, value: Any) -> Any:
        """Process a value before converting to plugin field.

        Args:
            name (str): Name of the field
            value (Any): Value to process

        Returns:
            Any: Processed value
        """
        if isinstance(value, IntEnum | IntFlag | trt.DataType):
            return value.value
        return value

    @classmethod
    def as_plugin_field(cls, name: str, value: Any) -> trt.PluginField:
        """Convert a value to a TensorRT plugin field.

        Args:
            name (str): Name of the field
            value (Any): Value to convert

        Returns:
            trt.PluginField: TensorRT plugin field
        """
        dtype = cls.get_field_dtype(name, value)
        value = cls.process_value(name, value)
        plugin_field_type = PLUGIN_FIELD_TYPES[dtype]
        return trt.PluginField(name, np.array(value, dtype=dtype), plugin_field_type)

    def get_fields(self) -> list[trt.PluginField]:
        """Get list of plugin fields from model attributes.

        Returns:
            list[trt.PluginField]: List of TensorRT plugin fields
        """
        return [self.as_plugin_field(name, value) for name, value in self.model_dump().items()]

    @property
    def __name__(self) -> str:
        """Get snake case name of plugin class.

        Returns:
            str: Snake case class name
        """
        return camel_to_snake(type(self).__name__)  # type: ignore[arg-type]

    def __hash__(self) -> int:
        """Get hash of plugin instance.

        Returns:
            int: Hash value combining plugin name and instance id
        """
        return hash(f"{self.__name__}_{id(self)}")

    def __eq__(self, other: object) -> bool:
        """Check if two plugin instances are equal.

        Args:
            other (object): Object to compare with

        Returns:
            bool: True if other is same plugin instance, False otherwise
        """
        if isinstance(other, Plugin):
            return self is other
        return False

    @abstractmethod
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply plugin operation to input tensors.

        This method is only required for fake tensor mode.

        Args:
            *args (Any): Positional arguments
            **kwargs (Any): Keyword arguments

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Output tensor(s)
        """


def camel_to_snake(name: str) -> str:
    """Convert camel case string to snake case.

    Args:
        name (str): Camel case string

    Returns:
        str: Snake case string
    """
    # First convert any sequence of uppercase letters to lowercase
    # except if followed by a lowercase letter
    name = re.sub(r"([A-Z]+)(?=[A-Z][a-z])", lambda m: m.group(1).lower(), name)
    # Then insert underscore before any remaining capitals and convert to lowercase
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
