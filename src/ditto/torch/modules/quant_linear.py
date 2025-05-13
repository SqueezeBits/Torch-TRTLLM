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

# pylint: disable=attribute-defined-outside-init, not-callable, too-many-instance-attributes

import torch

from ..ops import ditto_fake_quantize


class QuantLinear(torch.nn.Module):
    """A fake QuantLinear module.

    This module is used to quantize the input, weight, and output of a Linear module.
    """

    def __init__(self) -> None:
        super().__init__()

        self.is_input_quantized = False
        self.is_weight_quantized = False
        self.is_output_quantized = False

        self.input_num_bits = self.weight_num_bits = self.output_num_bits = 0
        self.input_dynamic = self.weight_dynamic = self.output_dynamic = False
        self.block_size: int | None = None
        self.register_buffer("weight", None)
        self.register_buffer("bias", None)
        self.register_buffer("weight_scale", None)
        self.register_buffer("weight_zero_point", None)
        self.register_buffer("input_scale", None)
        self.register_buffer("output_scale", None)

    def enable_weight_quantizer(
        self,
        weight: torch.Tensor,
        num_bits: int,
        dynamic: bool,
        *,
        block_size: int | None = None,
        scale: torch.Tensor | None = None,
        zero_point: torch.Tensor | None = None,
    ) -> None:
        """Enable weight quantization.

        Args:
            weight (torch.Tensor): The weight tensor.
            num_bits (int): The number of bits.
            dynamic (bool): Whether the quantization is dynamic.
            block_size (int | None): The block size. Defaults to None.
            scale (torch.Tensor | None): The scale tensor. Defaults to None.
            zero_point (torch.Tensor | None): The zero point tensor. Defaults to None.
        """
        self.is_weight_quantized = True
        self.weight = weight
        self.weight_num_bits = num_bits
        self.weight_dynamic = dynamic
        self.block_size = block_size
        self.weight_scale = scale
        self.weight_zero_point = zero_point

    def enable_input_quantizer(self, num_bits: int, dynamic: bool, *, scale: torch.Tensor | None = None) -> None:
        """Enable input quantization.

        Args:
            num_bits (int): The number of bits.
            dynamic (bool): Whether the quantization is dynamic.
            scale (torch.Tensor | None): The scale tensor. Defaults to None.
        """
        self.is_input_quantized = True
        self.input_num_bits = num_bits
        self.input_dynamic = dynamic
        self.input_scale = scale

    def enable_output_quantizer(self, num_bits: int, dynamic: bool, *, scale: torch.Tensor | None = None) -> None:
        """Enable output quantization.

        Args:
            num_bits (int): The number of bits.
            dynamic (bool): Whether the quantization is dynamic.
            scale (torch.Tensor | None): The scale tensor. Defaults to None.
        """
        self.is_output_quantized = True
        self.output_num_bits = num_bits
        self.output_dynamic = dynamic
        self.output_scale = scale

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inp (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if self.is_input_quantized:
            inp = ditto_fake_quantize(inp, self.input_num_bits, self.input_dynamic, inp.dtype, self.input_scale)

        if self.is_weight_quantized:
            assert self.weight_scale is not None, "Weight scale is required if weight quantization is enabled"
            weight = ditto_fake_quantize(
                self.weight,
                self.weight_num_bits,
                self.weight_dynamic,
                self.weight_scale.dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.block_size,
            ).T
        else:
            weight = self.weight

        out = torch.nn.functional.linear(
            inp, weight.to(inp.device), self.bias.to(inp.device) if self.bias is not None else None
        )

        if self.is_output_quantized:
            out = ditto_fake_quantize(out, self.output_num_bits, self.output_dynamic, out.dtype, self.output_scale)

        return out
