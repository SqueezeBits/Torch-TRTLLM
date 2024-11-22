# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from pydantic import model_validator
from torch.fx.node import Node
from typing_extensions import Self

from ....types import Number
from .aten_op import ATenOp


class Binary(ATenOp):
    this: Node | Number
    other: Node | Number

    @model_validator(mode="after")
    def check_if_one_of_the_inputs_is_node(self) -> Self:
        assert isinstance(self.this, Node) or isinstance(
            self.other, Node
        ), f"Expected one of the inputs of {self.node} to be a `torch.fx.Node` but got x={self.this}, y={self.other}"
        return self

    @property
    def is_commutative(self) -> bool:
        raise NotImplementedError(f"is_commutative for {type(self).__name__} is not implemented.")
