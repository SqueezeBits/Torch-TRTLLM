# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ...utils import get_tensor_metadata
from .aten_op import ATenOp
from .utils import make_dim_nonnegative


@ATenOp.final(torch.ops.aten.index_select.default)
class IndexSelectNode(ATenOp):
    this: Node
    dim: int
    index: Node

    @property
    def output_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
