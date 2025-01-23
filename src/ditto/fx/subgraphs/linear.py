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

import torch
from torch._subclasses import FakeTensor
from torch.fx import Graph, Node
from typing_extensions import Self

from ditto.fx.utils import get_val

from ..nodes import MM, AddTensorTensor, Permute, Reshape
from ..nodes.plugins import Gemm
from ..utils import get_ancestors_with_depth
from .subgraph import Subgraph


class Linear(Subgraph):
    """A subgraph representing a linear layer.

    This subgraph identifies a pattern of matrix multiplication with an optional bias addition,
    which is equivalent to a linear/dense layer in neural networks.
    The matrix multiplication node can be either a MM or a Gemm node.

    The layer performs: output = input @ weight.T + bias

    Attributes:
        mm (MM | Gemm): The matrix multiplication operation node
        add (AddTensor | None): The bias addition operation node, if present
    """

    mm: MM | Gemm
    add: AddTensorTensor | None

    @property
    def weight_node(self) -> Node:
        """The weight parameter node."""
        return self.mm.other

    @property
    def weight_tensor(self) -> FakeTensor:
        """The weight parameter tensor."""
        assert (weight := get_val(self.mm.other, FakeTensor)) is not None
        return weight

    @property
    def has_transposed_weight(self) -> bool:
        """Whether the weight is transposed."""
        if isinstance(self.mm, Gemm):
            return self.mm.target.transb
        return False

    @property
    def bias_node(self) -> Node | None:
        """The bias parameter node if present."""
        return self.add.other if self.add is not None else None

    @property
    def bias_tensor(self) -> FakeTensor | None:
        """The bias parameter tensor if present."""
        if self.add is not None:
            return get_val(self.add.other, FakeTensor)
        return None

    @property
    def has_transposed_input(self) -> bool:
        """Whether the input is transposed."""
        if isinstance(self.mm, Gemm):
            return self.mm.target.transa
        return False

    @property
    def input_node(self) -> Node:
        """The input tensor node to the linear layer."""
        if (
            isinstance(self.mm, Gemm)
            and self.has_transposed_input
            and (permute := Permute.specialize_from(self.mm.this)) is not None
        ):
            return permute.this
        return self.mm.this

    @property
    def output_node(self) -> Node:
        """The output tensor node, either the bias addition or matrix multiplication result."""
        return self.add.node if self.add is not None else self.mm.node

    @property
    def reshape_in(self) -> Reshape | None:
        """The reshape operation before the linear layer if present."""
        return Reshape.specialize_from(self.mm.this)

    @property
    def reshape_out(self) -> Reshape | None:
        """The reshape operation after the linear layer if present."""
        if len(users := list(self.output_node.users)) != 1:
            return None
        return Reshape.specialize_from(users[0])

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (mm := MM.specialize_from(node) or Gemm.specialize_from(node))
            and (weight := get_val(mm.other, torch.Tensor)) is not None
        ):
            return None

        add = AddTensorTensor.specialize_from(users[0]) if len(users := list(mm.users)) == 1 else None
        has_transposed_weight = isinstance(mm, Gemm) and mm.node.target.transb
        if add is not None and not (
            add.this == mm.node
            and (bias := get_val(add.other, torch.Tensor)) is not None
            and bias.shape[-1] == weight.shape[0 if has_transposed_weight else -1]
        ):
            add = None
        return cls(mm=mm, add=add)


def find_nearest_linear_projection(x: Node) -> Linear | None:
    """Find the nearest Linear projection subgraph by traversing up the node's ancestors.

    Searches through all ancestor nodes and finds the Linear projection subgraph that is closest
    to the given node in terms of graph traversal depth. This is useful for identifying the
    linear transformation that most directly affects the node's computation.

    Args:
        x: Starting node to search ancestors from

    Returns:
        The nearest Linear projection subgraph if one exists in the ancestors, None otherwise
    """
    if not (
        ancestor_linear_subgraphs := {
            subgraph: depth
            for node, depth in get_ancestors_with_depth(x).items()
            if (subgraph := Linear.configure_from(node))
        }
    ):
        return None
    return min(ancestor_linear_subgraphs, key=lambda subgraph: ancestor_linear_subgraphs[subgraph])


def find_last_linear(graph: Graph) -> Linear | None:
    """Find the last Linear subgraph in the computation graph.

    Args:
        graph (Graph): The computation graph to search in

    Returns:
        Linear | None: The last Linear subgraph if found, None otherwise
    """
    nodes = list(graph.nodes)
    for node in reversed(nodes):
        if subgraph := Linear.configure_from(node):
            return subgraph
    return None
