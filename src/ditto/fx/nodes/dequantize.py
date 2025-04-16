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

from collections.abc import Callable
from typing import Any

import torch
from torch.fx import Node

from ...quantization import QuantizeMode
from .call_function import FinalCallFunction
from .get_attr import GetAttr


class Dequantize(FinalCallFunction):
    """A representation of torch.ops.ditto.dequantize.default operation.

    Attributes:
        x (Node): The input tensor node.
        bits (int): The number of bits.
        dynamic (bool): Whether the quantization is dynamic.
        output_dtype (torch.dtype): The output data type.
        scale (Node | None): The scale tensor node. Defaults to None.
        zeros (Node | None): The unpacked zeros tensor node, if any. Defaults to None.
        group_size (int | None): The group size, if any. Defaults to None.
    """

    x: Node
    bits: int
    dynamic: bool
    output_dtype: torch.dtype
    scale: Node | None = None
    zeros: Node | None = None
    group_size: int | None = None

    @property
    def input_tensor(self) -> torch.Tensor | None:
        """Get the input tensor.

        Returns:
            torch.Tensor | None: The input tensor or None if not found
        """
        if attr := GetAttr.specialize_from(self.x):
            return attr.tensor
        return None

    @property
    def scale_tensor(self) -> torch.Tensor | None:
        """Get the scale tensor.

        Returns:
            torch.Tensor | None: The scale tensor or None if not found
        """
        if self.scale is not None and (attr := GetAttr.specialize_from(self.scale)):
            return attr.tensor
        return None

    @property
    def zeros_tensor(self) -> torch.Tensor | None:
        """Get the zeros tensor.

        Returns:
            torch.Tensor | None: The zeros tensor or None if not found
        """
        if self.zeros is not None and (attr := GetAttr.specialize_from(self.zeros)):
            return attr.tensor
        return None

    @property
    def quantize_mode(self) -> QuantizeMode:
        """Get the quantization mode.

        Returns:
            QuantizeMode: The quantization mode
        """
        if (scale := self.scale_tensor) is not None:
            if self.group_size is not None:
                return QuantizeMode.PER_GROUP
            if scale.ndim in (0, 1):
                return QuantizeMode.PER_TENSOR
            if scale.ndim == 2 and (scale.shape[1] == 1):
                return QuantizeMode.PER_CHANNEL
        if self.dynamic:
            return QuantizeMode.PER_TOKEN
        return QuantizeMode.UNKNOWN

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.ditto.dequantize.default,)
