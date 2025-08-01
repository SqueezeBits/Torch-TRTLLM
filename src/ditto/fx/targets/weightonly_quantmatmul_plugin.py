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

from enum import IntEnum
from typing import Any

import numpy as np
import tensorrt as trt
import torch

from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin


class WeightTypeId(IntEnum):
    """Type ID for weight tensor.

    Attributes:
        INT8 (int): Type ID for int8 weight tensor.
        INT4 (int): Type ID for int4 weight tensor.
    """

    INT8 = 1
    INT4 = 2


class WeightOnlyQuantMatmulPlugin(Plugin):
    """TensorRT plugin for matrix multiplication with weight-only quantization.

    Attributes:
        weight_type_id (WeightTypeId): Type ID for weight tensor.
        type_id (trt.DataType): Data type for computation.
    """

    weight_type_id: WeightTypeId
    type_id: trt.DataType

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for a plugin field value.

        Args:
            name (str): Name of the field
            value (Any): Value to get dtype for

        Returns:
            type[np.number]: numpy dtype for the value
        """
        if name == "weight_type_id":
            return np.int32
        return super().get_field_dtype(name, value)

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
            return torch.zeros(
                x.shape[0],
                weight.shape[1] if self.weight_type_id == WeightTypeId.INT8 else weight.shape[1] * 2,
                dtype=x.dtype,
            )
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
