# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from collections.abc import Callable
from typing import Any, Literal

import torch
from pydantic import model_validator
from torch.fx.node import Node
from typing_extensions import Self

from ...types import Number
from .call_function_node import CallFunctionNode
from .specialized_node import Asterick


class BinaryElementwiseNode(CallFunctionNode):
    x: Node | Number
    y: Node | Number

    @model_validator(mode="after")
    def check_if_one_of_the_inputs_is_node(self) -> Self:
        assert isinstance(self.x, Node) or isinstance(
            self.y, Node
        ), f"Expected one of the inputs of {self.node} to be a `torch.fx.Node` but got x={self.x}, y={self.y}"
        return self

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.pow.Scalar,
            torch.ops.aten.pow.Tensor_Scalar,
            torch.ops.aten.pow.Tensor_Tensor,
            torch.ops.aten.sub.Tensor,
        )

    @property
    def is_commutative(self) -> bool:
        raise NotImplementedError(f"is_commutative for {type(self).__name__} is not implemented.")


class BinaryElementwiseWithAlpha(BinaryElementwiseNode):
    asterick: None = Asterick
    alpha: Number = 1

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
        )


class AddNode(BinaryElementwiseWithAlpha):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.add.Tensor,)

    @property
    def is_commutative(self) -> Literal[True]:
        return True


class DivNode(BinaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.div.Tensor,)

    @property
    def is_commutative(self) -> Literal[False]:
        return False


class MulNode(BinaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.mul.Tensor,)

    @property
    def is_commutative(self) -> Literal[True]:
        return True


class PowNode(BinaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.pow.Scalar,
            torch.ops.aten.pow.Tensor_Scalar,
            torch.ops.aten.pow.Tensor_Tensor,
        )

    @property
    def is_commutative(self) -> Literal[False]:
        return False


class SubNode(BinaryElementwiseWithAlpha):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.sub.Tensor,)

    @property
    def is_commutative(self) -> Literal[False]:
        return False
