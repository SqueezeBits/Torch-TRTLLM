# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from .aten_op import ATenOp, FinalATenOp
from .utils import make_dim_nonnegative


@ATenOp.register(torch.ops.aten.index_select.default)
class IndexSelect(FinalATenOp):
    this: Node
    dim: int
    index: Node

    @property
    def output_ndim(self) -> int | None:
        if isinstance(t := self.output, torch.Tensor):
            return t.ndim
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
