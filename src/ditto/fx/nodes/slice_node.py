# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from ...types import SymInt
from ...utils import make_axis_nonnegative, make_dim_nonnegative
from ..utils import get_tensor_metadata
from .call_function_node import CallFunctionNode


class SliceNode(CallFunctionNode):
    x: Node
    dim: int = 0
    start: SymInt | None = None
    end: SymInt | None = None
    step: SymInt = 1

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.slice.Tensor,)

    @property
    def dim_size(self) -> int | None:
        if (t := get_tensor_metadata(self.x)) and isinstance(s := t.shape[self.dim], int):
            return s
        return None

    @property
    def ndim(self) -> int | None:
        if t := get_tensor_metadata(self.x):
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
