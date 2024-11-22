# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from collections.abc import Callable
from typing import Any

import torch
from pydantic import Field
from torch.fx.node import Node

from ...utils import make_dim_nonnegative
from ..utils import get_tensor_metadata
from .call_function_node import CallFunctionNode
from .specialized_node import Asterick


class ReductionIntListNode(CallFunctionNode):
    x: Node
    dim: list[int] = Field(max_length=1, min_length=1)
    keepdim: bool = False
    asterick: None = Asterick
    dtype: torch.dtype | None = None

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.mean.dim,
            torch.ops.aten.sum.dim_IntList,
        )

    @property
    def input_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.x):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim[0], ndim=ndim)
        return None


class MeanDimNode(ReductionIntListNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.mean.dim,)


class SumDimIntListNode(ReductionIntListNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.sum.dim_IntList,)
