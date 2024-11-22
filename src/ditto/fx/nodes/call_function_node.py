# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from .specialized_node import SpecializedNode


class CallFunctionNode(SpecializedNode):
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
