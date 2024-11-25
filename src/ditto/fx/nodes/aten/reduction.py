# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from pydantic import Field
from torch.fx.node import Node

from ...utils import get_tensor_metadata
from ..asterick import Asterick
from .aten_op import ATenOp
from .utils import make_dim_nonnegative


class Reduction(ATenOp):
    this: Node
    dim: list[int] = Field(max_length=1, min_length=1)
    keepdim: bool = False
    asterick: None = Asterick
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


@Reduction.final(torch.ops.aten.mean.dim)
class MeanDim(Reduction):
    ...


@Reduction.final(torch.ops.aten.sum.dim_IntList)
class SumDimIntList(Reduction):
    ...
