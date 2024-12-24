# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import SymbolicInteger
from ...utils import get_tensor_metadata
from .aten_op import ATenOp, FinalATenOp
from .utils import make_axis_nonnegative, make_dim_nonnegative


@ATenOp.register(torch.ops.aten.slice.Tensor)
class Slice(FinalATenOp):
    this: Node
    dim: int = 0
    start: SymbolicInteger | Node | None = None
    end: SymbolicInteger | Node | None = None
    step: SymbolicInteger | Node = 1

    @property
    def dim_size(self) -> int | None:
        if (t := get_tensor_metadata(self.this)) and isinstance(s := t.shape[self.dim], int):
            return s
        return None

    @property
    def ndim(self) -> int | None:
        if t := get_tensor_metadata(self.this):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None

    @property
    def nonnegative_start(self) -> int | None:
        if isinstance(self.start, int) and (dim_size := self.dim_size) is not None:
            return make_axis_nonnegative(self.start, dim_size=dim_size)
        return None

    @property
    def nonnegative_end(self) -> int | None:
        if isinstance(self.end, int) and (dim_size := self.dim_size) is not None:
            return make_axis_nonnegative(self.end, dim_size=dim_size)
        return None
