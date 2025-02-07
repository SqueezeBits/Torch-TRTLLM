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
    this: Node
    dim: int

    @property
    def input_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.this):
            return len(t.shape)
        return None

    @property
    def output_ndim(self) -> int | None:
        if isinstance(t := self.output, torch.Tensor):
            return t.ndim
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        return None


@ATenOp.register(torch.ops.aten.expand.default)
class Expand(FinalATenOp):
    this: Node
    shape: ShapeArg
    asterisk: None = Asterisk
    implicit: bool = False


@ATenOp.register(torch.ops.aten.permute.default)
class Permute(FinalATenOp):
    this: Node
    dims: list[int]

    @property
    def ndim(self) -> int:
        return len(self.dims)


@ATenOp.register(torch.ops.aten.reshape.default)
class Reshape(FinalATenOp):
    this: Node
    shape: ShapeArg


@ATenOp.register(torch.ops.aten.view.default)
class View(FinalATenOp):
    this: Node
    size: ShapeArg


@SingleDimensionReshape.register(torch.ops.aten.squeeze.dim)
class SqueezeDim(SingleDimensionReshape, FinalATenOp):
    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


@SingleDimensionReshape.register(torch.ops.aten.unsqueeze.default)
class Unsqueeze(SingleDimensionReshape, FinalATenOp):
    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
