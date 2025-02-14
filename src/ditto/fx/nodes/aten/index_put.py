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


@ATenOp.register(torch.ops.aten.index_put.default)
class IndexPut(FinalATenOp):
    """A node representing index_put operation.

    Attributes:
        this (Node): The input tensor to be modified
        indices (list[Node]): A list of tensors containing the indices to index
        values (Node): The tensor containing values to put into the input tensor
        accumulate (bool): Whether to accumulate values instead of replacing. Default: False
    """

    this: Node
    indices: list[Node]
    values: Node
    accumulate: bool = False
