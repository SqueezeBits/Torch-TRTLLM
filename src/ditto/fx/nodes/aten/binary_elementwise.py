# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from typing import Literal

import torch
from torch.fx import Node

from ....types import Number
from ..asterick import Asterick
from .aten_op import FinalATenOp
from .binary import Binary


class BinaryElementwise(Binary):
    ...


class BinaryElementwiseWithAlpha(BinaryElementwise):
    asterick: None = Asterick
    alpha: Number = 1


@BinaryElementwiseWithAlpha.register(torch.ops.aten.add.Tensor)
class Add(BinaryElementwiseWithAlpha, FinalATenOp):
    @property
    def is_commutative(self) -> Literal[True]:
        return True


@BinaryElementwise.register(torch.ops.aten.div.Tensor)
class Div(BinaryElementwise, FinalATenOp):
    @property
    def is_commutative(self) -> Literal[False]:
        return False


@BinaryElementwise.register(torch.ops.aten.mul.Tensor)
class Mul(BinaryElementwise, FinalATenOp):
    @property
    def is_commutative(self) -> Literal[True]:
        return True


class Pow(BinaryElementwise):
    @property
    def is_commutative(self) -> Literal[False]:
        return False


@Pow.register(torch.ops.aten.pow.Scalar)
class PowScalar(Pow, FinalATenOp):
    this: Number
    other: Node


@Pow.register(torch.ops.aten.pow.Tensor_Scalar)
class PowTensorScalar(Pow, FinalATenOp):
    this: Node
    other: Number


@Pow.register(torch.ops.aten.pow.Tensor_Tensor)
class PowTensorTensor(Pow, FinalATenOp):
    this: Node
    other: Node


@BinaryElementwiseWithAlpha.register(torch.ops.aten.sub.Tensor)
class Sub(BinaryElementwiseWithAlpha, FinalATenOp):
    @property
    def is_commutative(self) -> Literal[False]:
        return False
