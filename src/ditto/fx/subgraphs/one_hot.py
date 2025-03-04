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

from torch.fx import Node
from typing_extensions import Self

from ..nodes import ArangeStartStep, EqTensor
from .subgraph import Subgraph


class OneHot(Subgraph):
    """A subgraph representing a one-hot encoding operation.

    This subgraph identifies the pattern of operations used to create a one-hot encoded tensor
    by comparing an input tensor with a range of indices.

    Attributes:
        eq (EqTensor): The equality comparison operation node
        arange (ArangeStartStep): The arange operation that generates the index range
        this (Node): The input tensor to be one-hot encoded
    """

    eq: EqTensor
    arange: ArangeStartStep
    this: Node

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (eq := EqTensor.specialize_from(node))
            and (arange_and_other := eq.specialize_either_side_as(ArangeStartStep))
        ):
            return None

        arange, other = arange_and_other
        if not isinstance(other, Node):
            return None

        return cls(
            eq=eq,
            arange=arange,
            this=other,
        )
