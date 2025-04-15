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

# pylint: disable=unused-argument

import torch


@torch.library.custom_op("ditto::dequantize", mutates_args=())
def dequantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    zeros: torch.Tensor | None = None,
    group_size: int | None = None,
) -> torch.Tensor:
    """Dequantize the weight tensor.

    This function's implementation is not supported in the no fake mode.

    Args:
        weight (torch.Tensor): The weight tensor to dequantize.
        scale (torch.Tensor): The scale tensor.
        bits (int): The number of bits.
        zeros (torch.Tensor | None): The zeros tensor. Defaults to None.
        group_size (int | None): The group size. Defaults to None.

    Returns:
        torch.Tensor: The dequantized weight tensor.
    """
    raise NotImplementedError("Dequantization is not supported in the no fake mode.")


@torch.library.register_fake("ditto::dequantize")
def fake_dequantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    zeros: torch.Tensor | None = None,
    group_size: int | None = None,
) -> torch.Tensor:
    """Fake dequantize the weight tensor.

    This function is a fake version of the torch.ops.ditto.dequantize operation.

    Args:
        weight (torch.Tensor): The weight tensor to dequantize.
        scale (torch.Tensor): The scale tensor.
        bits (int): The number of bits.
        zeros (torch.Tensor | None): The zeros tensor. Defaults to None.
        group_size (int | None): The group size. Defaults to None.

    Returns:
        torch.Tensor: The dequantized weight tensor.
    """
    return torch.zeros_like(weight, dtype=scale.dtype, device=weight.device)


ditto_dequantize = torch.ops.ditto.dequantize
