# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from ...types import SymInt
from .call_function_node import CallFunctionNode


class SplitNode(CallFunctionNode):
    x: Node
    split_size: list[SymInt] | SymInt
    dim: int = 0

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.split.default, torch.ops.aten.split.sizes)
