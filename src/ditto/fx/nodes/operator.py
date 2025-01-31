import operator
from collections.abc import Callable
from typing import Any

from torch.fx import Node

from .call_function import FinalCallFunction


class GetItem(FinalCallFunction):
    """A specialization representing operator.getitem() nodes.

    Attributes:
        this (Node): The input sequence/container node to get item from
        idx (int): The index to retrieve
    """

    this: Node
    idx: int

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        """Get the possible target functions for getitem operation."""
        return (operator.getitem,)
