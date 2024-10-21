import logging
from abc import abstractmethod
from typing import TypeVar, overload

from pydantic import Field, ValidationError
from pydantic_core import PydanticUndefined
from torch.fx.node import Argument, Node
from typing_extensions import Self

from ..types import StrictlyTyped

logger = logging.getLogger(__name__)


class NodeArguments(StrictlyTyped):
    """Abstract base class for defining a specific node's arguments."""

    node: Node = Field(exclude=True)

    @classmethod
    @abstractmethod
    def validate_node(cls, node: Node) -> bool:
        """Validate if the node is suitable for argument extraction.

        Args:
            node (Node): a node

        Returns:
            bool: `True` if it is suitable, `False` otherwise.
        """
        raise NotImplementedError(f"{cls.__name__}.validate_node is not implemented")

    @classmethod
    def extract_from(cls, node: Node) -> Self | None:
        """Extract arguments from the node.

        Args:
            node (Node): a node

        Returns:
            Self | None: the extracted arguments if succeeded, `None` otherwise.
        """
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
        except ValidationError as e:
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
