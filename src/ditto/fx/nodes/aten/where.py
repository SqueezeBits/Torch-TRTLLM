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

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.where.self)
class Where(FinalATenOp):
    """The final specialization of `torch.ops.aten.where.self`.

    Attributes:
        condition (Node): The boolean mask tensor
        this (Node): The tensor to select from where condition is True
        other (Node): The tensor to select from where condition is False
    """

    condition: Node
    this: Node
    other: Node
