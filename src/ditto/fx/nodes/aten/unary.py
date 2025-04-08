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


class Unary(ATenOp):
    """Base class for unary operations.

    Attributes:
        this (Node): The input node.
    """

    this: Node


@Unary.register(torch.ops.aten.amin.default)
class AMin(Unary, FinalATenOp):
    """The final specialization of `torch.ops.aten.amin.default`.

    Attributes:
        dim (list[int]): The dimensions to reduce.
        keepdim (bool): Whether to keep the reduced dimensions.
    """

    dim: list[int]
    keepdim: bool = False


@Unary.register(torch.ops.aten.amax.default)
class AMax(Unary, FinalATenOp):
    """The final specialization of `torch.ops.aten.amax.default`.

    Attributes:
        dim (list[int]): The dimensions to reduce.
        keepdim (bool): Whether to keep the reduced dimensions.
    """

    dim: list[int]
    keepdim: bool = False


@Unary.register(torch.ops.aten.aminmax.default)
class AMinMax(Unary, FinalATenOp):
    """The final specialization of `torch.ops.aten.aminmax.default`.

    Attributes:
        dim (int | None): The dimension to reduce.
        keepdim (bool): Whether to keep the reduced dimension.
    """

    dim: int | None = None
    keepdim: bool = False
