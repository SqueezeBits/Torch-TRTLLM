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

from collections.abc import Sequence

import torch
from torch.fx.node import Node
from typing_extensions import Self

from ....types import SymbolicInteger
from ...utils import get_tensor_metadata
from .aten_op import ATenOp, FinalATenOp
from .utils import has_same_values, make_axis_nonnegative, make_dim_nonnegative


@ATenOp.register(torch.ops.aten.slice.Tensor)
class Slice(FinalATenOp):
    """Specialization for the slice operator.

    Attributes:
        this (Node): The tensor to slice.
        dim (int): The dimension to slice along.
        start (SymbolicInteger | Node | None): The start of the slice.
        end (SymbolicInteger | Node | None): The end of the slice.
        step (SymbolicInteger | Node): The step of the slice.
    """

    this: Node
    dim: int = 0
    start: SymbolicInteger | Node | None = None
    end: SymbolicInteger | Node | None = None
    step: SymbolicInteger | Node = 1

    @classmethod
    def sort(cls, slices: Sequence[Self]) -> list[Self]:
        """Sort the slices by the start of the slice.

        Args:
            slices (Sequence[Self]): The slices to sort.

        Returns:
            list[Self]: The sorted slices.
        """
        return sorted(slices, key=lambda s: start if isinstance(start := s.start, int) else 0)

    @classmethod
    def are_consecutive(cls, slices: Sequence[Self]) -> bool:
        """Check if a sequence of slice operations are consecutive.

        Slices are considered consecutive if they:
        1. Operate on the same tensor
        2. Have the same dimension size
        3. All have valid dimension sizes (no None values)
        4. All have step size of 1
        5. Each slice's end matches the next slice's start
        6. The final slice's end matches the original dimension size

        Args:
            slices (Sequence[Self]): A sequence of Slice operations to check

        Returns:
            bool: True if the slices are consecutive, False otherwise
        """
        return (
            len({s.this for s in slices}) == 1
            and len(dim_sizes := {s.dim_size for s in slices}) == 1
            and None not in dim_sizes
            and all(s.step == 1 for s in slices)
            and all(
                has_same_values(slices[i].nonnegative_end, slices[i + 1].nonnegative_start)
                for i in range(len(slices) - 1)
            )
            and has_same_values(slices[-1].nonnegative_end, slices[0].dim_size)
        )

    @property
    def dim_size(self) -> int | None:
        """The size of the dimension to slice along.

        Returns:
            int | None: The size of the dimension to slice along.
        """
        if (t := get_tensor_metadata(self.this)) and isinstance(s := t.shape[self.dim], int):
            return s
        return None

    @property
    def ndim(self) -> int | None:
        """The number of dimensions of the tensor.

        Returns:
            int | None: The number of dimensions of the tensor.
        """
        if t := get_tensor_metadata(self.this):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        """The non-negative dimension of the tensor.

        Returns:
            int | None: The non-negative dimension of the tensor.
        """
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None

    @property
    def nonnegative_start(self) -> int | None:
        """The non-negative start of the slice.

        Returns:
            int | None: The non-negative start of the slice.
        """
        if isinstance(self.start, int) and (dim_size := self.dim_size) is not None:
            return make_axis_nonnegative(self.start, dim_size=dim_size)
        return None

    @property
    def nonnegative_end(self) -> int | None:
        """The non-negative end of the slice.

        Returns:
            int | None: The non-negative end of the slice.
        """
        if isinstance(self.end, int) and (dim_size := self.dim_size) is not None:
            return make_axis_nonnegative(self.end, dim_size=dim_size)
        return None
