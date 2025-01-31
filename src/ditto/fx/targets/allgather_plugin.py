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


class AllGatherPlugin(Plugin):
    """TensorRT plugin implementation of All Gather.

    Attributes:
        group (list[int]): The group of ranks to gather from
        type_id (trt.DataType): The data type of the tensor to gather
    """

    # the order of the attributes does matter!
    group: list[int]
    type_id: trt.DataType

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if is_in_fake_tensor_mode():
            # plugin gathers as 1D flattened tensor
            # [dim0, ..., dimN] -> [group_size * dim0, ..., dimN]
            return x.repeat(len(self.group), 1)
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")
