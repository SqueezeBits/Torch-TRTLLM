# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar, overload

import torch
import torch.utils._pytree as pytree
from loguru import logger
from pydantic import Field, ValidationError
from pydantic_core import PydanticUndefined
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Graph
from torch.fx.node import Argument, Node, Target
from typing_extensions import Self

from ...types import ShapeArg, StrictlyTyped, SymbolicShape
from ..utils import find_sym_size_node


class NodeSpecialization(StrictlyTyped, ABC):
    """Base class for specializing FX nodes with type-safe validation.

    Attributes:
        node (Node): The underlying FX node being specialized
    """

    node: Node = Field(exclude=True, frozen=True)

    @classmethod
    @abstractmethod
    def designated_op(
        cls,
    ) -> Literal["call_function", "call_method", "call_module", "get_attr", "placeholder", "output",]:
        """Get the FX operation type that this specialization is designated for."""

    @property
    def op(self) -> str:
        """The operation type of the underlying node."""
        return self.node.op

    @property
    def target(self) -> Target:
        """The target of the underlying node."""
        return self.node.target

    @property
    def users(self) -> dict[Node, None]:
        """The users of the underlying node."""
        return self.node.users

    @property
    def name(self) -> str:
        """The name of the underlying node."""
        return self.node.name

    @property
    def stack_trace(self) -> str | None:
        """The stack trace of the underlying node."""
        return self.node.stack_trace

    @stack_trace.setter
    def stack_trace(self, value: str | None) -> None:
        """Set the stack trace string for this node."""
        self.node.stack_trace = value

    @property
    def output(self) -> FakeTensor | torch.SymInt | None:
        """The output value of this node."""
        return self.node.meta.get("val")

    @output.setter
    def output(self, other: FakeTensor | torch.SymInt | None) -> None:
        """Set the output value of this node."""
        self.node.meta["val"] = other

    @property
    def output_shape(self) -> SymbolicShape | None:
        """The shape of the output value of this node."""
        if isinstance(t := self.output, FakeTensor):
            return (*t.shape,)
        if isinstance(self.output, torch.SymInt):
            return ()
        return None

    @property
    def output_shape_arg(self) -> ShapeArg | None:
        """The shape of the output value of this node that can be provided as an argument for creating other nodes."""
        if (shape := self.output_shape) is None:
            return None

        try:
            return [s if isinstance(s, int) else find_sym_size_node(self.node.graph, s) for s in shape]
        except RuntimeError as e:
            logger.warning(e)

        return None

    @property
    def output_dtype(self) -> torch.dtype | None:
        """The dtype of the output value of this node."""
        if isinstance(t := self.output, FakeTensor):
            return t.dtype
        if isinstance(self.output, torch.SymInt):
            return torch.int64
        return None

    def args_kwargs(self, **hotfix: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Extract arguments and keyword arguments from this node's fields.

        Args:
            **hotfix: Override field values before extraction

        Returns:
            tuple: A tuple containing (args, kwargs) extracted from fields
        """
        _args: list[Any] = []
        _kwargs: dict[str, Any] = {}
        append_in_args = True
        data = self.model_dump()
        if hotfix:
            data.update(hotfix)
        for name, field in self.model_fields.items():
            if field.exclude:
                if name == "asterick":
                    append_in_args = False
                continue
            value = data[name]
            if field.is_required() or value != field.get_default(call_default_factory=True):
                if append_in_args:
                    _args.append(value)
                else:
                    _kwargs[name] = value
            if append_in_args and not field.is_required():
                append_in_args = False
        return tuple(_args), _kwargs

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        """Validate if the node is suitable for argument extraction.

        Args:
            node (Node): a node

        Returns:
            bool: `True` if it is suitable, `False` otherwise.
        """
        return node.op == cls.designated_op()

    @classmethod
    def specialize_from(cls, node: Node) -> Self | None:
        """Specialize from the given node.

        Args:
            node (Node): a node

        Returns:
            Self | None: the specialized node if succeeded, `None` otherwise.
        """
        try:
            return cls._specialize_from(node)
        except (AssertionError, TypeError, ValidationError):
            return None

    @classmethod
    def _specialize_from(cls, node: Node) -> Self:
        """Specialize from the given node unsafely.

        Args:
            node (Node): The node to specialize from

        Returns:
            Self: The specialized node

        Raises:
            TypeError: If node validation fails
            AssertionError: If this class has further validation logic
            ValidationError: If the `node.args` and `node.kwargs` do not match the expected structure
        """
        if not cls.validate_node(node):
            raise TypeError(f"{node.format_node()} cannot be validated as {cls.__name__}")
        arguments = {
            name: get_argument(
                node,
                index - 1,
                name,
                default=None if field.default is PydanticUndefined else field.default,
            )
            for index, (name, field) in enumerate(cls.model_fields.items())
            if index > 0  # should skip the `node`
        }
        return cls.model_validate({"node": node, **arguments})

    @classmethod
    def unwrap_specialization(
        cls,
        *args: "Argument | NodeSpecialization",
        **kwargs: "Argument | NodeSpecialization",
    ) -> tuple[tuple[Argument, ...], dict[str, Argument]]:
        """Unwrap specialized nodes back to their underlying nodes.

        Args:
            *args: Positional arguments that may contain specialized nodes
            **kwargs: Keyword arguments that may contain specialized nodes

        Returns:
            tuple: A tuple containing unwrapped (args, kwargs)
        """
        flat_args, spec = pytree.tree_flatten((args, kwargs))
        return pytree.tree_unflatten(
            (x.node if isinstance(x, NodeSpecialization) else x for x in flat_args),
            spec,
        )

    def __str__(self) -> str:
        return self.node.format_node()


class FinalSpecialization(NodeSpecialization):
    """Base class for final node specializations that can be instantiated."""

    @classmethod
    @abstractmethod
    def create(
        cls,
        graph: Graph,
        *args: Argument | NodeSpecialization,
        **kwargs: Argument | NodeSpecialization,
    ) -> Self:
        """Create a new node of this specialization type.

        Args:
            graph (Graph): The FX graph to create the node in
            *args: Positional arguments for the node
            **kwargs: Keyword arguments for the node

        Returns:
            Self: The created specialized node
        """


DefaultValue = TypeVar("DefaultValue")


@overload
def get_argument(
    node: Node,
    index_as_arg: int,
    name_as_kwarg: str,
    default: DefaultValue,
) -> DefaultValue:
    ...


@overload
def get_argument(
    node: Node,
    index_as_arg: int,
    name_as_kwarg: str,
) -> Argument:
    ...


def get_argument(
    node: Node,
    index_as_arg: int,
    name_as_kwarg: str,
    default: DefaultValue | None = None,
) -> DefaultValue | Argument:
    """Get the node argument of the given node.

    Args:
        node (Node): a node
        index_as_arg (int): the index to look up when the node argument is given as a positional argument
        name_as_kwarg (str): the key to look up when the node argument is given as a keyword argument
        default (DefaultValue | None, optional): the default value when the node argument is not explicitly specified.
            Defaults to None.

    Returns:
        DefaultValue | Argument: the node argument if found or its default value.
    """
    return (
        node.kwargs[name_as_kwarg]
        if name_as_kwarg in node.kwargs
        else (node.args[index_as_arg] if len(node.args) > index_as_arg else default)
    )
