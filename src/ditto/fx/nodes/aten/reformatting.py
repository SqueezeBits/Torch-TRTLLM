# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import SymInt
from ...utils import get_tensor_metadata
from .aten_op import ATenOp
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
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        return None


@ATenOp.final(torch.ops.aten.permute.default)
class Permute(ATenOp):
    this: Node
    dims: list[int]

    @property
    def ndim(self) -> int:
        return len(self.dims)


@ATenOp.final(torch.ops.aten.reshape.default)
class Reshape(ATenOp):
    this: Node
    shape: list[SymInt]

    @property
    def target_shape(self) -> torch.Size | None:
        sym_ints: list[int | torch.SymInt] = []
        for s in self.shape:
            if isinstance(s, int | torch.SymInt):
                sym_ints.append(s)
                continue
            if not isinstance(val := s.meta.get("val"), torch.SymInt):
                return None
            sym_ints.append(val)
        return torch.Size(sym_ints)  # type: ignore[arg-type]


@SingleDimensionReshape.final(torch.ops.aten.squeeze.dim)
class SqueezeDim(SingleDimensionReshape):
    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


@SingleDimensionReshape.final(torch.ops.aten.unsqueeze.default)
class Unsqueeze(SingleDimensionReshape):
    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
