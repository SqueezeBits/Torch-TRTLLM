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
