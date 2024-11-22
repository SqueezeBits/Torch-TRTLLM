# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from typing import Literal

import torch

from ....types import Number
from ..asterick import Asterick
from .binary import Binary


class BinaryElementwise(Binary):
    ...


class BinaryElementwiseWithAlpha(BinaryElementwise):
    asterick: None = Asterick
    alpha: Number = 1


@BinaryElementwiseWithAlpha.final(torch.ops.aten.add.Tensor)
class Add(BinaryElementwiseWithAlpha):
    @property
    def is_commutative(self) -> Literal[True]:
        return True


@BinaryElementwise.final(torch.ops.aten.div.Tensor)
class Div(BinaryElementwise):
    @property
    def is_commutative(self) -> Literal[False]:
        return False


@BinaryElementwise.final(torch.ops.aten.mul.Tensor)
class Mul(BinaryElementwise):
    @property
    def is_commutative(self) -> Literal[True]:
        return True


@BinaryElementwise.final(
    torch.ops.aten.pow.Scalar,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.pow.Tensor_Tensor,
)
class Pow(BinaryElementwise):
    @property
    def is_commutative(self) -> Literal[False]:
        return False


@BinaryElementwiseWithAlpha.final(torch.ops.aten.sub.Tensor)
class Sub(BinaryElementwiseWithAlpha):
    @property
    def is_commutative(self) -> Literal[False]:
        return False
