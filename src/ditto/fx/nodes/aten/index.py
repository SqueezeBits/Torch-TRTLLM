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

from ...nodes.get_attr import GetAttr
from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.index.Tensor)
class Index(FinalATenOp):
    """The final specialization of `torch.ops.aten.index.Tensor`.

    This class represents the specialized behavior of the `torch.ops.aten.index.Tensor`
    operation in a computational graph. It provides properties to determine if the
    operation can be replaced with a single slicing operation and to extract indexing
    details.

    Attributes:
        this (Node): The node representing the `torch.ops.aten.index.Tensor` operation.
        indices (tuple[Node | None, ...]): The tuple of index nodes or None values
            representing the indices used in the operation.
    """

    this: Node
    indices: tuple[Node | None, ...]

    @property
    def can_replace_with_single_slice(self) -> bool:
        """Check if the operation can be replaced with a single slicing operation.

        This property verifies that the indexing operation uses exactly one index,
        which is derived from a single-element tensor.
        """
        return (
            len(self.node.all_input_nodes) == 2
            and (constant := GetAttr.specialize_from(self.node.all_input_nodes[1]))
            and isinstance(constant.parameter, torch.Tensor)
            and constant.parameter.numel() == 1
        )

    @property
    def dim(self) -> int:
        """The dimension along which the slicing operation would occur.

        Raises an assertion error if `can_replace_with_single_slice` is False.
        """
        assert self.can_replace_with_single_slice
        return self.indices.index(self.node.all_input_nodes[-1])

    @property
    def idx(self) -> int:
        """The resolved index value for the slicing operation, accounting for negative indices.

        Raises an assertion error if `can_replace_with_single_slice` is False.
        """
        assert self.can_replace_with_single_slice

        idx_tensor = GetAttr.specialize_from(self.node.all_input_nodes[-1]).parameter
        idx = int(idx_tensor.item())

        if idx < 0 and "val" in self.this.meta and isinstance(dim_size := self.this.meta["val"].size(self.dim), int):
            idx = dim_size + idx

        return idx
