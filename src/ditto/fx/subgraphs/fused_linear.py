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

from ..nodes import Slice
from .linear import Linear
from .subgraph import Subgraph


class FusedLinear(Subgraph):
    """A subgraph representing a fused linear layer whose output is split into multiple parts.

    This subgraph identifies a pattern of a linear layer followed by slice operations that split
    its output into consecutive chunks. This is commonly used in attention layers where a single
    linear projects to query, key and value tensors that are then sliced apart.

    Attributes:
        linear (Linear): The linear layer subgraph
        slices (tuple[Slice, ...]): The slice operations that split the linear output,
            sorted in order of their slice ranges
    """

    linear: Linear
    slices: tuple[Slice, ...]

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (linear := Linear.configure_from(node)) is None:
            return None
        output = linear.reshape_out or linear.output_node
        if not (
            all(Slice.specialize_from(user) is not None for user in output.users)
            and Slice.are_consecutive(slices := Slice.sort([Slice._specialize_from(user) for user in output.users]))
        ):
            return None
        return cls(linear=linear, slices=tuple(slices))

    def __len__(self) -> int:
        return len(self.slices)
