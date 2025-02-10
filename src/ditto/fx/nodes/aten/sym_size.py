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
from torch.fx import Node

from ...utils import get_tensor_metadata
from .aten_op import ATenOp, FinalATenOp
from .utils import make_dim_nonnegative


@ATenOp.register(torch.ops.aten.sym_size.int)
class SymSizeInt(FinalATenOp):
    """Specialization for the sym_size operator with an integer dimension.

    Attributes:
        this (Node): The input node.
        dim (int): The dimension to get the size of.
    """

    this: Node
    dim: int

    @property
    def ndim(self) -> int | None:
        """The number of dimensions of the input tensor.

        Returns:
            int | None: The number of dimensions of the input tensor.
        """
        if t := get_tensor_metadata(self.this):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        """The non-negative dimension of the input tensor.

        Returns:
            int | None: The non-negative dimension of the input tensor.
        """
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
