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

# pylint: disable=unused-argument,too-many-positional-arguments

import torch


@torch.library.custom_op("ditto::fake_quantize", mutates_args=())
def fake_quantize(
    x: torch.Tensor,
    bits: int,
    dynamic: bool,
    output_dtype: torch.dtype,
    scale: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
    group_size: int | None = None,
    smoother: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake-quantize the input tensor.

    This function's implementation is not supported in the no fake mode.

    Args:
        x (torch.Tensor): The input tensor to fake-quantize.
        bits (int): The number of bits.
        dynamic (bool): Whether the quantization is dynamic.
        output_dtype (torch.dtype | None): The output data type.
        scale (torch.Tensor | None): The scale tensor. Defaults to None.
        zeros (torch.Tensor | None): The zeros tensor. Defaults to None.
        group_size (int | None): The group size. Defaults to None.
        smoother (torch.Tensor | None): The smoothing factor tensor. Defaults to None.

    Returns:
        torch.Tensor: The fake-quantized weight tensor.
    """
    raise NotImplementedError("ditto::fake_quantize is not supported in the no fake mode.")


@torch.library.register_fake("ditto::fake_quantize")
def _(
    x: torch.Tensor,
    bits: int,
    dynamic: bool,
    output_dtype: torch.dtype,
    scale: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
    group_size: int | None = None,
    smoother: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake ditto::fake_quantize the input tensor.

    This function is a fake version of the torch.ops.ditto.fake_quantize operation.

    Args:
        x (torch.Tensor): The input tensor to fake-quantize.
        bits (int): The number of bits.
        dynamic (bool): Whether the quantization is dynamic.
        output_dtype (torch.dtype | None): The output data type.
        scale (torch.Tensor | None): The scale tensor. Defaults to None.
        zeros (torch.Tensor | None): The zeros tensor. Defaults to None.
        group_size (int | None): The group size. Defaults to None.
        smoother (torch.Tensor | None): The smoothing factor tensor. Defaults to None.

    Returns:
        torch.Tensor: The fake-quantized weight tensor.
    """
    return torch.zeros_like(x, dtype=output_dtype, device=x.device)


ditto_fake_quantize = torch.ops.ditto.fake_quantize
