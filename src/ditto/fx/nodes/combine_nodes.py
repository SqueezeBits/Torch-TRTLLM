# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from ...utils import make_dim_nonnegative
from ..utils import get_tensor_metadata
from .call_function_node import CallFunctionNode


class CombineNode(CallFunctionNode):
    tensors: list[Node]
    dim: int = 0

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.cat.default,
            torch.ops.aten.stack.default,
        )


class CatNode(CombineNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.cat.default,)

    @property
    def ndim(self) -> int | None:
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


class StackNode(CombineNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.stack.default,)
