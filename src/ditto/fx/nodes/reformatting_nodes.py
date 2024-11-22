# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from ...types import SymInt
from ...utils import make_dim_nonnegative
from ..utils import get_tensor_metadata
from .call_function_node import CallFunctionNode


class SingleDimensionReshape(CallFunctionNode):
    x: Node
    dim: int

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.unsqueeze.default,
        )

    @property
    def input_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.x):
            return len(t.shape)
        return None

    @property
    def output_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None


class PermuteNode(CallFunctionNode):
    x: Node
    dims: list[int]

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.permute.default,)

    @property
    def ndim(self) -> int:
        return len(self.dims)


class ReshapeNode(CallFunctionNode):
    x: Node
    shape: list[SymInt]

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.reshape.default,)

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


class SqueezeDimNode(SingleDimensionReshape):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.squeeze.dim,)

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


class UnsqueezeNode(SingleDimensionReshape):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.unsqueeze.default,)

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
