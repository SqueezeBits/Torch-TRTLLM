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

from ....types import Number
from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.clamp.default)
class ClampScalar(FinalATenOp):
    """The final specialization of `torch.ops.aten.clamp.default`.

    Attributes:
        this (Node): The tensor operand
        min (Number): The minimum value
        max (Number): The maximum value
    """

    this: Node
    min: Number | None = None
    max: Number | None = None


@ATenOp.register(torch.ops.aten.clamp.Tensor)
class ClampTensor(FinalATenOp):
    """The final specialization of `torch.ops.aten.clamp.Tensor`.

    Attributes:
        this (Node): The tensor operand
        min (Tensor | None): The minimum value
        max (Tensor | None): The maximum value
    """

    this: Node
    min: Node | None = None
    max: Node | None = None
