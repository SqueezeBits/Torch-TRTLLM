# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar, overload

from loguru import logger
from pydantic import Field, ValidationError
from pydantic_core import PydanticUndefined
from torch.fx.node import Argument, Node, Target
from typing_extensions import Self

from ...types import StrictlyTyped


class NodeSpecialization(StrictlyTyped, ABC):
    """Abstract base class for defining node whose arguments are specialized for a specific op and target(s)."""

    node: Node = Field(exclude=True, frozen=True)

    @classmethod
    @abstractmethod
    def designated_op(
        cls,
    ) -> Literal["call_function", "call_method", "call_module", "get_attr", "placeholder", "output",]:
        ...

    @classmethod
    @abstractmethod
    def possible_targets(cls) -> tuple[Target, ...]:
        ...

    @property
    def op(self) -> str:
        return self.node.op

    @property
    def target(self) -> Target:
        return self.node.target

    @property
    def name(self) -> str:
        return self.node.name

    def args_kwargs(self, **hotfix: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
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
            if append_in_args:
                _args.append(value)
            elif field.is_required() or value != field.get_default(call_default_factory=True):
                _kwargs[name] = value
        return tuple(_args), _kwargs

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        """Validate if the node is suitable for argument extraction.

        Args:
            node (Node): a node

        Returns:
            bool: `True` if it is suitable, `False` otherwise.
        """
        return node.op == cls.designated_op() and node.target in cls.possible_targets()

    @classmethod
    def specialize_from(cls, node: Node) -> Self | None:
        """Specialize from the given node.

        Args:
            node (Node): a node

        Returns:
            Self | None: the specialized node if succeeded, `None` otherwise.
        """
        assert len(cls.possible_targets()) > 0, f"{cls.__name__} does not have possible targets"
        if not cls.validate_node(node):
            return None
        try:
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
        except (AssertionError, ValidationError) as e:
            logger.warning(f"Incorrect arguments given to the node {node.format_node()}: {arguments}. ({e})")
            return None


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
