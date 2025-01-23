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

from .aten_op import ATenOp, FinalATenOp
from .utils import make_dim_nonnegative


class Combine(ATenOp):
    tensors: list[Node]
    dim: int = 0


@Combine.register(torch.ops.aten.cat.default)
class Cat(Combine, FinalATenOp):
    @property
    def ndim(self) -> int | None:
        if isinstance(t := self.output, torch.Tensor):
            return t.ndim
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


@Combine.register(torch.ops.aten.stack.default)
class Stack(Combine, FinalATenOp):
    ...
