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

from abc import abstractmethod

from torch.fx import Node
from typing_extensions import Self

from ..nodes import Mul, Sigmoid
from .subgraph import Subgraph


class ActivationSubgraph(Subgraph):
    """Base class for activation subgraphs."""

    @property
    @abstractmethod
    def input(self) -> Node:
        """The input node of the activation subgraph."""

    @property
    @abstractmethod
    def output(self) -> Node:
        """The output node of the activation subgraph."""


class Silu(ActivationSubgraph):
    """The SiLU subgraph.

    Args:
        sigmoid (Sigmoid): The sigmoid node.
        mul (Mul): The mul node.
    """

    sigmoid: Sigmoid
    mul: Mul

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (
            (mul := Mul.specialize_from(node))
            and isinstance(rhs := mul.other, Node)
            and (sigmoid := Sigmoid.specialize_from(rhs))
            and sigmoid.this == mul.this
        ):
            return cls(sigmoid=sigmoid, mul=mul)
        return None

    @property
    def input(self) -> Node:
        return self.sigmoid.this

    @property
    def output(self) -> Node:
        return self.mul.node
