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

from ....types import Number, SymbolicInteger
from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.topk.default)
class TopK(FinalATenOp):
    """Specialization for the topk operator.

    Attributes:
        this (Node): The input tensor.
        k (SymbolicInteger): Number of top elements to return.
        dim (Number): The dimension to sort along. Default: -1 (last dimension).
        largest (bool): Whether to return the largest or smallest elements. Default: True.
        sorted (bool): Whether to return the elements in sorted order. Default: True.
    """

    this: Node
    k: SymbolicInteger
    dim: Number = -1
    largest: bool = True
    sorted: bool = True
