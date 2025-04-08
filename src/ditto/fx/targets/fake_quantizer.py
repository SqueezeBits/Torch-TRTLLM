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

from ...quantization import GlobalQuantConfig, QuantizeMode, QuantizeType
from ...types import StrictlyTyped
from .fake_tensor_mode import is_in_fake_tensor_mode


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


class Dequantizer(StrictlyTyped):
    """Fake dequantization target.

    This target wraps the subgraph that holds the quantization information of the linear layer
    into a single dequantize target. It is used to convert the mm node to a plugin node for quantization.

    Attributes:
        dtype (torch.dtype): Data type of the model
        global_quant_config (GlobalQuantConfig): Global quantization configuration
        output_shape (torch.Size): Shape of the output tensor
        bits (int): Number of bits for quantization
        mode (QuantizeMode): Mode of quantization
        group_size (int | None): Size of quantization groups. Defaults to None.
    """

    dtype: torch.dtype
    global_quant_config: GlobalQuantConfig
    output_shape: torch.Size
    bits: int
    mode: QuantizeMode
    group_size: int | None = None

    @property
    def __name__(self) -> str:
        """Get the name of the class.

        Returns:
            str: Name of the class
        """
        return "dequantizer"

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
        return isinstance(other, Dequantizer) and self is other

    def __call__(
        self,
        x_q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply node operation to input tensors.

        This method is only required for fake tensor mode.

        Args:
            x_q (torch.Tensor): Quantized input tensor
            scale (torch.Tensor): Scale tensor
            zero_point (torch.Tensor | None): Zero point tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if is_in_fake_tensor_mode():
            return torch.zeros(self.output_shape, dtype=self.dtype)
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")
