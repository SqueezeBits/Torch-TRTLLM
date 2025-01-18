from collections.abc import Callable
from typing import Any

from torch.fx.node import Node

from ..nodes import CallFunction
from ..targets import GemmPlugin


class Gemm(CallFunction):
    """A plugin specialization representing a gemm plugin node.

    Attributes:
        this (Node): The first input node
        other (Node): The second input node (expected to be a weight tensor)
    """

    this: Node
    other: Node

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, GemmPlugin)
