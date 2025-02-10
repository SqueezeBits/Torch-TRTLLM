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

from ....types import ShapeArg
from ...utils import get_tensor_metadata
from ..asterisk import Asterisk
from .aten_op import ATenOp, FinalATenOp
from .utils import make_dim_nonnegative


class SingleDimensionReshape(ATenOp):
    """Base class for nodes that perform single-dimension reshaping operations.

    Attributes:
        this (Node): The tensor to reshape.
        dim (int): The dimension to reshape.
    """

    this: Node
    dim: int

    @property
    def input_ndim(self) -> int | None:
        """The number of dimensions of the input tensor.

        Returns:
            int | None: The number of dimensions of the input tensor.
        """
        if t := get_tensor_metadata(self.this):
            return len(t.shape)
        return None

    @property
    def output_ndim(self) -> int | None:
        """The number of dimensions of the output tensor.

        Returns:
            int | None: The number of dimensions of the output tensor.
        """
        if isinstance(t := self.output, torch.Tensor):
            return t.ndim
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        """The non-negative dimension of the tensor.

        Returns:
            int | None: The non-negative dimension of the tensor.
        """
        return None


@ATenOp.register(torch.ops.aten.expand.default)
class Expand(FinalATenOp):
    """Specialization for the expand operator.

    Attributes:
        this (Node): The tensor to expand.
        shape (ShapeArg): The shape to expand the tensor to.
        asterisk (None): The asterisk of the expand.
        implicit (bool): Whether the expand is implicit.
    """

    this: Node
    shape: ShapeArg
    asterisk: None = Asterisk
    implicit: bool = False


@ATenOp.register(torch.ops.aten.permute.default)
class Permute(FinalATenOp):
    """Specialization for the permute operator.

    Attributes:
        this (Node): The tensor to permute.
        dims (list[int]): The dimensions to permute the tensor to.
    """

    this: Node
    dims: list[int]

    @property
    def ndim(self) -> int:
        """The number of dimensions of the tensor.

        Returns:
            int: The number of dimensions of the tensor.
        """
        return len(self.dims)


@ATenOp.register(torch.ops.aten.reshape.default)
class Reshape(FinalATenOp):
    """Specialization for the reshape operator.

    Attributes:
        this (Node): The tensor to reshape.
        shape (ShapeArg): The shape to reshape the tensor to.
    """

    this: Node
    shape: ShapeArg


@ATenOp.register(torch.ops.aten.view.default)
class View(FinalATenOp):
    """Specialization for the view operator.

    Attributes:
        this (Node): The tensor to view.
        size (ShapeArg): The size to view the tensor to.
    """

    this: Node
    size: ShapeArg


@SingleDimensionReshape.register(torch.ops.aten.squeeze.dim)
class SqueezeDim(SingleDimensionReshape, FinalATenOp):
    """Specialization for the squeeze operator."""

    @property
    def nonnegative_dim(self) -> int | None:
        """The non-negative dimension of the tensor.

        Returns:
            int | None: The non-negative dimension of the tensor.
        """
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


@SingleDimensionReshape.register(torch.ops.aten.unsqueeze.default)
class Unsqueeze(SingleDimensionReshape, FinalATenOp):
    """Specialization for the unsqueeze operator."""

    @property
    def nonnegative_dim(self) -> int | None:
        """The non-negative dimension of the tensor.

        Returns:
            int | None: The non-negative dimension of the tensor.
        """
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
