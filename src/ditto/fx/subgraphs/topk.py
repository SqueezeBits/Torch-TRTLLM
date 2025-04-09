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

from ...types import SymbolicInteger
from ..nodes import GetItem
from ..nodes import TopK as TopKNode
from .subgraph import Subgraph


class TopK(Subgraph):
    """A subgraph representing a top-k operation.

    This subgraph identifies the components of a top-k operation, including the input tensor
    and optional output nodes for values and indices.

    Attributes:
        this (Node): The input tensor node for the top-k operation.
        output_values (GetItem | None): Node representing the output values from top-k.
            None if values are not used.
        output_indices (GetItem | None): Node representing the output indices from top-k.
            None if indices are not used.
    """

    this: Node
    k: SymbolicInteger
    output_values: GetItem | None
    output_indices: GetItem | None

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        """Configure a TopK subgraph starting from a given node.

        Args:
            node (Node): The starting node for pattern matching, expected to be a GetItem operation.

        Returns:
            TopK | None: A configured TopK instance if the pattern matches, None otherwise.
        """
        if not ((get_item := GetItem.specialize_from(node)) and (topk := TopKNode.specialize_from(get_item.this))):
            return None

        assert len(topk.users) in (1, 2)
        if len(topk.users) == 1:
            values = get_item if get_item.idx == 0 else None
            indices = get_item if get_item.idx == 1 else None
        else:
            values, indices = (GetItem.specialize_from(user) for user in topk.users)

        assert any(get_item == user for user in (values, indices))
        return cls(
            this=topk.this,
            k=topk.k,
            output_values=values,
            output_indices=indices,
        )
