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

import numpy as np
import tensorrt as trt
import torch

from ...types import DataType
from .plugin import Plugin


class Fp8RowwiseGemmPlugin(Plugin):
    """TensorRT plugin for Fp8RowwiseGemm.

    Attributes:
        has_per_channel_scaling (bool): Whether to use per-channel scaling. Defaults to False.
        has_per_token_scaling (bool): Whether to use per-token scaling. Defaults to False.
        type_id (trt.DataType): Data type for the model.
    """

    # the order of the attributes does matter!
    has_per_channel_scaling: bool
    has_per_token_scaling: bool
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
        if name in ("has_per_channel_scaling", "has_per_token_scaling"):
            return np.int32
        return super().get_field_dtype(name, value)

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale_tokens: torch.Tensor,
        scale_channels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform matrix multiplication between input tensors.

        Args:
            x (torch.Tensor): First input tensor
            weight (torch.Tensor): Second input tensor (weight matrix)
            scale_tokens (torch.Tensor): Scaling factor for per-token scaling
            scale_channels (torch.Tensor): Scaling factor for per-channel scaling
            **kwargs (Any): Additional keyword arguments

        Returns:
            torch.Tensor: Result of matrix multiplication
        """
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=DataType(self.type_id).to(torch.dtype))
