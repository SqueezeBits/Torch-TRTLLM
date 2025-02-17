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

# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from .aten_op import ATenOp, FinalATenOp


class Softmax(ATenOp):
    """Base class for softmax operations.

    Attributes:
        this (Node): The input node.
        dim (int): The dimension to apply the softmax to.
    """

    this: Node
    dim: int


@Softmax.register(torch.ops.aten._softmax.default)
class SoftmaxDefault(Softmax, FinalATenOp):
    """Specialization for the softmax operator.

    Attributes:
        half_to_float (bool): Whether to convert the input to float.
    """

    half_to_float: bool


@Softmax.register(torch.ops.aten._safe_softmax.default)
class SafeSoftmax(Softmax, FinalATenOp):
    """Specialization for the safe softmax operator.

    Attributes:
        dtype (torch.dtype | None): The dtype of the safe softmax.
    """

    dtype: torch.dtype | None = None
