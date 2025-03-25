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
import torch

from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin


class WeightOnlyQuantMatmulPlugin(Plugin):
    """TensorRT plugin for matrix multiplication with weight-only quantization.

    Attributes:
        weight_type_id (trt.DataType): Type ID for weight tensor.
        type_id (trt.DataType): Data type for computation.
    """

    weight_type_id: trt.DataType
    type_id: trt.DataType

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform matrix multiplication between input tensors.

        Args:
            x (torch.Tensor): First input tensor
            weight (torch.Tensor): Second input tensor (weight matrix)
            scale (torch.Tensor): Scaling factor for quantization
            **kwargs (Any): Additional keyword arguments

        Returns:
            torch.Tensor: Result of matrix multiplication
        """
        if is_in_fake_tensor_mode():
            return torch.zeros(x.shape[0], scale.shape[1], dtype=x.dtype)
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")


class WeightOnlyGroupwiseQuantMatmulPlugin(Plugin):
    """TensorRT plugin for matrix multiplication with weight-only groupwise quantization.

    Attributes:
        type_id (trt.DataType): Data type for computation.
        quant_algo (int): Quantization algorithm.
        group_size (int): Size of quantization groups.
    """

    type_id: trt.DataType
    quant_algo: int
    group_size: int

    # pylint: disable=too-many-positional-arguments
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zeros: torch.Tensor | None = None,
        biases: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform matrix multiplication between input tensors.

        Args:
            x (torch.Tensor): First input tensor
            weight (torch.Tensor): Second input tensor (weight matrix)
            scale (torch.Tensor): Scaling factor for quantization
            zeros (torch.Tensor | None): Zeros tensor
            biases (torch.Tensor | None): Biases tensor
            alpha (torch.Tensor | None): Alpha tensor
            **kwargs (Any): Additional keyword arguments

        Returns:
            torch.Tensor: Result of matrix multiplication
        """
        if is_in_fake_tensor_mode():
            return torch.zeros(x.shape[0], scale.shape[1], dtype=x.dtype)
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")
