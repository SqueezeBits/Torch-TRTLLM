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


@ATenOp.register(torch.ops.aten.addmm.default)
class AddMM(FinalATenOp):
    """Specialization for the addmm operator.

    Attributes:
        this (Node): The tensor to add.
        mat1 (Node): The matrix to multiply.
        mat2 (Node): The matrix to multiply.
        beta (Number): The beta of the addmm.
        alpha (Number): The alpha of the addmm.
    """

    this: Node
    mat1: Node
    mat2: Node
    beta: Number = 1
    alpha: Number = 1
