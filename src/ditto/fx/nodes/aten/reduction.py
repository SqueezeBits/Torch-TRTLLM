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
from pydantic import Field
from torch.fx.node import Node

from ...utils import get_tensor_metadata
from ..asterisk import Asterisk
from .aten_op import ATenOp, FinalATenOp
from .utils import make_dim_nonnegative


class Reduction(ATenOp):
    this: Node
    dim: list[int] = Field(max_length=1, min_length=1)
    keepdim: bool = False
    asterisk: None = Asterisk
    dtype: torch.dtype | None = None

    @property
    def input_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.this):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim[0], ndim=ndim)
        return None


@Reduction.register(torch.ops.aten.mean.dim)
class MeanDim(Reduction, FinalATenOp):
    ...


@Reduction.register(torch.ops.aten.sum.dim_IntList)
class SumDimIntList(Reduction, FinalATenOp):
    ...
