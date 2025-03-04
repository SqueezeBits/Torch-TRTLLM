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

from ....types import SymbolicInteger
from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.select.int)
class SelectInt(FinalATenOp):
    """A node specialization for torch.ops.aten.select.int operation.

    Attributes:
        this (Node): The input tensor node to select from
        dim (int): The dimension along which to select
        index (SymbolicInteger): The index to select at the specified dimension
    """

    this: Node
    dim: int
    index: SymbolicInteger
