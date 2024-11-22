# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from ...utils import make_dim_nonnegative
from ..utils import get_tensor_metadata
from .call_function_node import CallFunctionNode


class IndexSelectNode(CallFunctionNode):
    x: Node
    dim: int
    index: Node

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.index_select.default,)

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
