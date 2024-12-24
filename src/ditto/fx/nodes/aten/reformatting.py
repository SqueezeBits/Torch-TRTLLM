# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import ShapeArg
from ...utils import get_tensor_metadata
from ..asterick import Asterick
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
    asterick: None = Asterick
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
