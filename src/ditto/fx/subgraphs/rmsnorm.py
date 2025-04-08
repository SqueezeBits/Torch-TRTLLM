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

# import torch
from torch.fx import Node
from typing_extensions import Self

from ..nodes import (
    AddTensorScalar,
    DivScalarTensor,
    GetAttr,
    MeanDim,
    MulTensorTensor,
    NativeLayerNorm,
    PowTensorScalar,
    Sqrt,
    ToCopy,
)
from .subgraph import Subgraph


class RmsNormSubgraph(Subgraph):
    """A RMSNorm layer subgraph.

    Attributes:
        top_node (Node): The top node of the subgraph.
        mul (Node): The bottom node of the subgraph.
        weight (GetAttr): The GetAttr specialization for the weight of the RMSNorm.
        bias (GetAttr | None): The GetAttr specialization for the bias of the RMSNorm.
        eps (float): The eps value of the RMSNorm.
    """

    input_node: Node
    mul: Node
    weight: GetAttr
    bias: GetAttr | None
    eps: float

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (
            (native_layer_norm := NativeLayerNorm.specialize_from(node))
            and isinstance(native_layer_norm.eps, float)
            and (
                weight := GetAttr.specialize_from(native_layer_norm.weight)
                and (bias := GetAttr.specialize_from(native_layer_norm.bias))
            )
        ):
            return cls(
                input_node=native_layer_norm.this,
                mul=node,
                weight=weight,
                bias=bias,
                eps=native_layer_norm.eps,
            )

        if not (
            (mul := MulTensorTensor.specialize_from(node))
            and (weight := GetAttr.specialize_from(mul.this) or GetAttr.specialize_from(mul.other))
            and (to_copy := ToCopy.specialize_from(mul.this) or ToCopy.specialize_from(mul.other))
            and (mul2 := MulTensorTensor.specialize_from(to_copy.this))
            and (div := DivScalarTensor.specialize_from(mul2.this) or DivScalarTensor.specialize_from(mul2.other))
            and (sqrt := Sqrt.specialize_from(div.other))
            and (add := AddTensorScalar.specialize_from(sqrt.this))
            and isinstance(add.other, float)
            and (mean := MeanDim.specialize_from(add.this))
            and (pow_tensor_scalar := PowTensorScalar.specialize_from(mean.this))
            and (to_copy2 := ToCopy.specialize_from(pow_tensor_scalar.this))
            and (mul2.node in list(to_copy2.users))
        ):
            return None

        return cls(input_node=to_copy2.this, mul=mul.node, weight=weight, bias=None, eps=add.other)
