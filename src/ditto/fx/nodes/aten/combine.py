# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ...utils import get_tensor_metadata
from .aten_op import ATenOp
from .utils import make_dim_nonnegative


class Combine(ATenOp):
    tensors: list[Node]
    dim: int = 0


@Combine.final(torch.ops.aten.cat.default)
class Cat(Combine):
    @property
    def ndim(self) -> int | None:
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


@Combine.final(torch.ops.aten.stack.default)
class Stack(Combine):
    ...
