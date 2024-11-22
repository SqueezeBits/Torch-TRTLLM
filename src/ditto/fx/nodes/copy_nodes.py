# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from .call_function_node import CallFunctionNode
from .specialized_node import Asterick


class CloneNode(CallFunctionNode):
    x: Node
    asterick: None = Asterick
    memory_format: torch.memory_format | None = None

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.clone.default,)


class ToCopyNode(CallFunctionNode):
    x: Node
    asterick: None = Asterick
    dtype: torch.dtype | None = None
    layout: torch.layout | None = None
    device: torch.device | None = None
    pin_memory: bool | None = None
    non_blocking: bool = False
    memory_format: torch.memory_format | None = None

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten._to_copy.default,)
