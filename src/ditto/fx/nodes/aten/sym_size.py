# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx import Node

from ...utils import get_tensor_metadata
from .aten_op import ATenOp, FinalATenOp
from .utils import make_dim_nonnegative


@ATenOp.register(torch.ops.aten.sym_size.int)
class SymSizeInt(FinalATenOp):
    this: Node
    dim: int

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
