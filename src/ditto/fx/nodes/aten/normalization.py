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


@ATenOp.register(torch.ops.aten.native_layer_norm.default)
class NativeLayerNorm(FinalATenOp):
    """Specialization for the native_layer_norm operator.

    Attributes:
        this (Node): The input tensor.
        normalized_shape (SymbolicInteger): The shape of the normalized shape.
        weight (Node): The weight tensor.
        bias (Node): The bias tensor.
        eps (Number): The epsilon value.
    """

    this: Node
    normalized_shape: SymbolicInteger
    weight: Node
    bias: Node
    eps: Number
