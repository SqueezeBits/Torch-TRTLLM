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

# pylint: disable=too-many-positional-arguments

from typing import Any

import numpy as np
import tensorrt as trt
import torch
from tensorrt_llm.functional import QuantMode

from ...types import DataType
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin


class RmsnormQuantizationPlugin(Plugin):
    """TensorRT plugin implementation of RmsnormQuantization.

    Attributes:
        eps (float): Epsilon value for RMS normalization
        dyn_act_scaling (bool): Whether to use dynamic activation scaling
        sum_per_token (bool): Whether to sum per token
        clamp_enabled (bool): Whether to clamp the output
        quant_mode (QuantMode): Quantization mode
        type_id (trt.DataType): Data type of the input tensor
        out_type_id (trt.DataType): Data type of the output tensor
    """

    # the order of the attributes does matter!
    eps: float
    dyn_act_scaling: bool
    sum_per_token: bool
    clamp_enabled: bool
    quant_mode: QuantMode
    type_id: trt.DataType
    out_type_id: trt.DataType

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for a plugin field value.

        Args:
            name (str): Name of the field
            value (Any): Value to get dtype for

        Returns:
            type[np.number]: numpy dtype for the value
        """
        if name in ("dyn_act_scaling", "sum_per_token", "clamp_enabled"):
            return np.int32
        return super().get_field_dtype(name, value)

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        scale: torch.Tensor,
        clamp_val: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        if is_in_fake_tensor_mode():
            output = [torch.zeros_like(x, dtype=DataType(self.out_type_id).to(torch.dtype))]
            if self.dyn_act_scaling:
                output.append(torch.zeros((x.shape[0], 1), dtype=torch.float32))
            if self.sum_per_token:
                output.append(torch.zeros((x.shape[0], 1), dtype=torch.float32))
            return tuple(output)
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")
