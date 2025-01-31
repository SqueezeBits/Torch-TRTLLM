from collections.abc import Callable
from typing import Any

from torch.fx.node import Node

from ..targets import GemmPlugin, GPTAttentionPlugin
from .call_function import CallFunction


class Gemm(CallFunction):
    """A plugin specialization representing a gemm plugin node.

    Attributes:
        this (Node): The first input node
        other (Node): The second input node (expected to be a weight tensor)
    """

    this: Node
    other: Node

    @property
    def target(self) -> GemmPlugin:
        assert isinstance(t := super().target, GemmPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, GemmPlugin)


class GPTAttention(CallFunction):
    """A plugin specialization representing a GPT attention plugin node."""

    model_config = {"extra": "ignore"}

    qkv: Node

    @property
    def target(self) -> GPTAttentionPlugin:
        assert isinstance(t := super().target, GPTAttentionPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, GPTAttentionPlugin)
