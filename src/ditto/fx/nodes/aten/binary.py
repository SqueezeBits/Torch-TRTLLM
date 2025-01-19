# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from abc import abstractmethod
from typing import TypeVar

from pydantic import model_validator
from torch.fx.node import Node
from typing_extensions import Self

from ....types import Number
from ..node_specialization import NodeSpecialization
from .aten_op import ATenOp

SomeSpecialization = TypeVar("SomeSpecialization", bound=NodeSpecialization)


class Binary(ATenOp):
    """Base class for binary ATen operators.

    Attributes:
        this (Node | Number): The first operand of the binary operation
        other (Node | Number): The second operand of the binary operation
    """

    this: Node | Number
    other: Node | Number

    @model_validator(mode="after")
    def check_if_one_of_the_inputs_is_node(self) -> Self:
        """Validate that at least one of the binary operation inputs is a Node.

        This validator ensures that either `this` or `other` is a `torch.fx.Node`. Having at least
        one Node input is required since we are operating within the FX graph system.

        Returns:
            Self: Returns self if validation passes

        Raises:
            AssertionError: If neither input is a Node
        """
        assert isinstance(self.this, Node) or isinstance(
            self.other, Node
        ), f"Expected one of the inputs of {self.node} to be a `torch.fx.Node` but got x={self.this}, y={self.other}"
        return self

    @property
    @abstractmethod
    def is_commutative(self) -> bool:
        """Whether the binary operation is commutative.

        Returns:
            bool: True if the operation is commutative, False otherwise
        """

    def specialize_either_side_as(self, node_type: type[SomeSpecialization]) -> SomeSpecialization | None:
        """Try to specialize either operand as a specific node type.

        Args:
            node_type (type[SomeSpecialization]): The node type to specialize as

        Returns:
            SomeSpecialization | None: The specialized node if successful, None otherwise
        """
        if isinstance(self.this, Node) and (lhs := node_type.specialize_from(self.this)):
            return lhs
        if isinstance(self.other, Node) and (rhs := node_type.specialize_from(self.other)):
            return rhs
        return None
