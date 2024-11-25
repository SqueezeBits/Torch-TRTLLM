# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from torch.fx.node import Node

from .node_specialization import NodeSpecialization


class CallFunction(NodeSpecialization):
    @property
    def target(self) -> Callable[..., Any]:
        assert callable(op := super().target)
        return op

    @classmethod
    def designated_op(cls) -> Literal["call_function"]:
        return "call_function"

    @classmethod
    @abstractmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        ...

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return super().validate_node(node) and node.target in cls.possible_targets()
