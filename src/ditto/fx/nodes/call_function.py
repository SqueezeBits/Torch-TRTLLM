# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from torch.fx import Graph
from torch.fx.node import Argument, Node
from typing_extensions import Self

from .node_specialization import FinalSpecialization, NodeSpecialization


class CallFunction(NodeSpecialization):
    """Base class for specializing call_function nodes.

    Provides common functionality for handling nodes that call Python functions.
    """

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
        """Get the possible function targets this specialization can handle.

        Returns:
            tuple[Callable[..., Any], ...]: Tuple of callable targets supported by this specialization
        """

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return super().validate_node(node) and node.target in cls.possible_targets()


class FinalCallFunction(CallFunction, FinalSpecialization):
    """Final specialization for call_function nodes that have exactly one target function."""

    @classmethod
    def designated_target(cls) -> Callable[..., Any]:
        """Get the single designated function target for this specialization.

        Returns:
            Callable[..., Any]: The single function target

        Raises:
            AssertionError: If there is not exactly one possible target
        """
        assert (
            len(targets := cls.possible_targets()) == 1
        ), f"Final ATen op must have exactly one target, but {cls.__name__} has {len(targets)} {targets = }"
        return targets[0]

    @classmethod
    def create(
        cls,
        graph: Graph,
        *args: Argument | NodeSpecialization,
        **kwargs: Argument | NodeSpecialization,
    ) -> Self:
        args_, kwargs_ = cls.unwrap_specialization(*args, **kwargs)
        node = graph.call_function(cls.designated_target(), args_, kwargs_)
        return cls._specialize_from(node)
