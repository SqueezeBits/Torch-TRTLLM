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

import torch
from pydantic import Field

from ...quantization import QuantizeMode, QuantizeType
from ...types import StrictlyTyped


class ActivationQuantization(StrictlyTyped):
    """Activation quantization."""

    quant_mode: QuantizeMode
    bits: int
    type: QuantizeType
    scale: torch.Tensor | None = Field(default=None, exclude=True)
    zero_point: torch.Tensor | None = Field(default=None, exclude=True)
    dynamic: bool = False


class Quantizer(StrictlyTyped):
    """Quantization target."""

    @property
    def __name__(self) -> str:
        """Get the name of the class.

        Returns:
            str: Name of the class
        """
        return "quantizer"

    def __hash__(self) -> int:
        """Get the hash of the class.

        Returns:
            int: Hash of the class
        """
        return hash(f"{self.__name__}_{id(self)}")

    def __eq__(self, other: object) -> bool:
        """Check if the class is equal to another object.

        Args:
            other (object): Another object

        Returns:
            bool: True if the class is equal to another object, False otherwise
        """
        return isinstance(other, Quantizer) and self is other

    def __call__(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply node operation to input tensors.

        This method is only required for fake tensor mode.

        Args:
            x (torcyh.Tensor): Input tensor
            scale (torch.Tensor): Scale tensor
            output_dtype (torch.dtype): Output data type
        Returns:
            torch.Tensor: Output tensor
        """
        out = x * scale
        return out.to(output_dtype)
