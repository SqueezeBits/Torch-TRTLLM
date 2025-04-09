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

from .plugin import Plugin


class TopkLastDimPlugin(Plugin):
    """Plugin that performs topk operation on the last dimension.

    Attributes:
        is_largest (bool): Whether to return largest elements (True) or smallest (False).
                          Defaults to True.
        k (int): Number of top elements to return
        type_id (trt.DataType): Data type of the input tensor
    """

    is_largest: bool = True
    k: int
    type_id: trt.DataType

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return x[..., : self.k]

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for a plugin field value.

        Args:
            name (str): Name of the field
            value (Any): Value to get dtype for

        Returns:
            type[np.number]: numpy dtype for the value
        """
        if name in ("is_largest"):
            return np.int32
        return super().get_field_dtype(name, value)
