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

from typing import Literal

import torch
from torch.fx.node import Node

from .aten_op import FinalATenOp
from .binary import Binary


@Binary.register(torch.ops.aten.mm.default)
class MM(Binary, FinalATenOp):
    """The final specialization of `torch.ops.aten.mm.default`.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
    """

    this: Node
    other: Node

    @property
    def is_commutative(self) -> Literal[False]:
        return False


@Binary.register(torch.ops.aten.bmm.default)
class BMM(Binary, FinalATenOp):
    """The final specialization of `torch.ops.aten.bmm.default`.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
    """

    this: Node
    other: Node

    @property
    def is_commutative(self) -> Literal[False]:
        return False
