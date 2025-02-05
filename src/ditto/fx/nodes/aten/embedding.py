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

from ....types import SymbolicInteger
from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.embedding.default)
class Embedding(FinalATenOp):
    """Specialization for the embedding operator.

    Attributes:
        weight (Node): The weight of the embedding.
        indices (Node): The indices of the embedding.
        padding_idx (SymbolicInteger): The padding index of the embedding.
        scale_grad_by_freq (bool): Whether to scale the gradient by the frequency of the indices.
        sparse (bool): Whether to use sparse gradient.
    """

    weight: Node
    indices: Node
    padding_idx: SymbolicInteger = -1
    scale_grad_by_freq: bool = False
    sparse: bool = False
