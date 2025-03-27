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
from tensorrt_llm.functional import QuantMode

from ...types import DataType
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin


class QuantizePerTokenPlugin(Plugin):
    """TensorRT plugin implementation of QuantizePerToken.

    Attributes:
        type_id (trt.DataType): Data type of the input tensor
        quant_mode (QuantMode): Quantization mode
        clamp_enabled (bool): Whether to clamp the output
        sum_per_token (bool): Whether to sum per token
    """

    # the order of the attributes does matter!
    type_id: trt.DataType
    quant_mode: QuantMode
    clamp_enabled: bool
    sum_per_token: bool

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for a plugin field value.

        Args:
            name (str): Name of the field
            value (Any): Value to get dtype for

        Returns:
            type[np.number]: numpy dtype for the value
        """
        if name in ("sum_per_token"):
            return np.int32
        return super().get_field_dtype(name, value)

    def __call__(
        self,
        x: torch.Tensor,
        clamp_val: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        if is_in_fake_tensor_mode():
            output = [
                torch.zeros_like(x, dtype=DataType(self.type_id).to(torch.dtype)),
                torch.zeros((x.shape[0], 1), dtype=torch.float32),
            ]
            if self.sum_per_token:
                output.append(torch.zeros((x.shape[0], 1), dtype=torch.float32))
            return tuple(output)
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")
