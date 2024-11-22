# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from .call_function_node import CallFunctionNode


class UnaryElementwiseNode(CallFunctionNode):
    x: Node

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.sqrt.default,
        )


class SqrtNode(UnaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.sqrt.default,)
