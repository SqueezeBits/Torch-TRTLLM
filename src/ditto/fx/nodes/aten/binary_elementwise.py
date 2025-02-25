# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from typing import Literal

import torch
from torch.fx import Node

from ....types import Number
from ..asterisk import Asterisk
from .aten_op import FinalATenOp
from .binary import Binary


class BinaryElementwise(Binary):
    """Base class for binary elementwise operations like add, mul, div etc.

    Attributes:
        this (Node | Number): The first operand
        other (Node | Number): The second operand
    """


class Add(BinaryElementwise):
    """Binary elementwise addition operation.

    Attributes:
        this (Node | Number): The first operand
        other (Node | Number): The second operand
    """

    @property
    def is_commutative(self) -> Literal[True]:
        return True


@Add.register(torch.ops.aten.add.Tensor)
class AddTensor(Add, FinalATenOp):
    """The final specialization of `torch.ops.aten.add.Tensor`.

    Adds a tensor or scalar with another tensor.

    Attributes:
        this (Node | Number): The first operand
        other (Node): The second tensor operand
        asterisk (None): Placeholder indicating mandatory keyword argument
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this + alpha * other`. Defaults to 1.
    """

    other: Node
    asterisk: None = Asterisk
    alpha: Number = 1


class AddScalarTensor(AddTensor):
    """A further specialization of `torch.ops.aten.add.Tensor`.

    Adds a scalar with a tensor.

    Attributes:
        this (Number): The first operand
        other (Node): The tensor to add
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this + alpha * other`. Defaults to 1.
    """

    this: Number


class AddTensorTensor(AddTensor):
    """A further specialization of `torch.ops.aten.add.Tensor`.

    Adds a tensor with another tensor.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this + alpha * other`. Defaults to 1.
    """

    this: Node


@Add.register(torch.ops.aten.add.Scalar)
class AddScalar(Add, FinalATenOp):
    """The final specialization of `torch.ops.aten.add.Scalar`.

    Adds a tensor or scalar with a scalar.

    Attributes:
        this (Node | Number): The first operand
        other (Number): The second scalar operand
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this + alpha * other`. Defaults to 1.
    """

    other: Number
    alpha: Number = 1


class AddTensorScalar(AddScalar):
    """A further specialization of `torch.ops.aten.add.Scalar`.

    Adds a tensor with a scalar.

    Attributes:
        this (Node): The first tensor operand
        other (Number): The second scalar operand
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this + alpha * other`. Defaults to 1.
    """

    this: Node


@Add.register(torch.ops.aten.add.default)
class AddDefault(Add, FinalATenOp):
    """The final specialization of `torch.ops.aten.add.default`.

    Adds a scalar with another scalar.

    Attributes:
        this (Number): The first scalar operand
        other (Number): The second scalar operand
    """

    this: Number
    other: Number


class Div(BinaryElementwise):
    """Binary elementwise division operation.

    Attributes:
        this (Node | Number): The first operand
        other (Node | Number): The second operand
    """

    @property
    def is_commutative(self) -> Literal[False]:
        return False


@Div.register(torch.ops.aten.div.Tensor)
class DivTensor(Div, FinalATenOp):
    """The final specialization of `torch.ops.aten.div.Tensor`.

    Divides a tensor or scalar by a tensor.

    Attributes:
        this (Node | Number): The first operand
        other (Node): The second tensor operand
    """

    other: Node


class DivScalarTensor(DivTensor):
    """A further specialization of `torch.ops.aten.div.Tensor`.

    Divides a scalar by a tensor.

    Attributes:
        this (Number): The first scalar operand
        other (Node): The second tensor operand
    """

    this: Number


class DivTensorTensor(DivTensor):
    """A further specialization of `torch.ops.aten.div.Tensor`.

    Divides a tensor by another tensor.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
    """

    this: Node


@Div.register(torch.ops.aten.div.Tensor_mode)
class DivTensorMode(Div, FinalATenOp):
    """The final specialization of `torch.ops.aten.div.Tensor_mode`.

    Divides a tensor or scalar by a tensor with rounding mode.

    Attributes:
        this (Node | Number): The first operand
        other (Node): The second tensor operand
        mode (Literal["floor", "trunc"] | None): The rounding mode
    """

    other: Node
    mode: Literal["floor", "trunc"] | None


@Div.register(torch.ops.aten.div.Scalar)
class DivScalar(Div, FinalATenOp):
    """The final specialization of `torch.ops.aten.div.Scalar`.

    Divides a tensor or scalar by a scalar.

    Attributes:
        this (Node | Number): The first operand
        other (Number): The second scalar operand
    """

    other: Number


class DivTensorScalar(DivScalar):
    """A further specialization of `torch.ops.aten.div.Scalar`.

    Divides a tensor by a scalar.

    Attributes:
        this (Node): The first tensor operand
        other (Number): The second scalar operand
    """

    this: Node


@Div.register(torch.ops.aten.div.Scalar_mode)
class DivScalarMode(Div, FinalATenOp):
    """The final specialization of `torch.ops.aten.div.Scalar_mode`.

    Divides a tensor or scalar by a scalar with rounding mode.

    Attributes:
        this (Node | Number): The first operand
        other (Number): The second scalar operand
        mode (Literal["floor", "trunc"] | None): The rounding mode
    """

    other: Number
    mode: Literal["floor", "trunc"] | None


@Div.register(torch.ops.aten.div.default)
class DivDefault(Div, FinalATenOp):
    """The final specialization of `torch.ops.aten.div.default`.

    Divides a scalar by another scalar.

    Attributes:
        this (Number): The first scalar operand
        other (Number): The second scalar operand
    """

    this: Number
    other: Number


class Mul(BinaryElementwise):
    """Binary elementwise multiplication operation.

    Attributes:
        this (Node | Number): The first operand
        other (Node | Number): The second operand
    """

    @property
    def is_commutative(self) -> Literal[True]:
        return True


@Mul.register(torch.ops.aten.mul.Tensor)
class MulTensor(Mul, FinalATenOp):
    """The final specialization of `torch.ops.aten.mul.Tensor`.

    Multiplies a tensor or scalar with a tensor.

    Attributes:
        this (Node | Number): The first operand
        other (Node): The second tensor operand
    """

    other: Node


class MulScalarTensor(MulTensor):
    """A further specialization of `torch.ops.aten.mul.Tensor`.

    Multiplies a scalar with a tensor.

    Attributes:
        this (Number): The first scalar operand
        other (Node): The second tensor operand
    """

    this: Number


class MulTensorTensor(MulTensor):
    """A further specialization of `torch.ops.aten.mul.Tensor`.

    Multiplies a tensor with another tensor.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
    """

    this: Node


@Mul.register(torch.ops.aten.mul.Scalar)
class MulScalar(Mul, FinalATenOp):
    """The final specialization of `torch.ops.aten.mul.Scalar`.

    Multiplies a tensor or scalar with a scalar.

    Attributes:
        this (Node | Number): The first operand
        other (Number): The second scalar operand
    """

    other: Number


class MulTensorScalar(MulScalar):
    """A further specialization of `torch.ops.aten.mul.Scalar`.

    Multiplies a tensor with a scalar.

    Attributes:
        this (Node): The first tensor operand
        other (Number): The second scalar operand
    """

    this: Node


@Mul.register(torch.ops.aten.mul.default)
class MulDefault(Mul, FinalATenOp):
    """The final specialization of `torch.ops.aten.mul.default`.

    Multiplies a scalar with another scalar.

    Attributes:
        this (Number): The first scalar operand
        other (Number): The second scalar operand
    """

    this: Number
    other: Number


class Pow(BinaryElementwise):
    """Binary elementwise power operation.

    Attributes:
        this (Node | Number): The first operand
        other (Node | Number): The second operand
    """

    @property
    def is_commutative(self) -> Literal[False]:
        return False


@Pow.register(torch.ops.aten.pow.Scalar)
class PowScalar(Pow, FinalATenOp):
    """The final specialization of `torch.ops.aten.pow.Scalar`.

    Raises a scalar to a tensor power.

    Attributes:
        this (Number): The first scalar operand
        other (Node): The second tensor operand
    """

    this: Number
    other: Node


@Pow.register(torch.ops.aten.pow.Tensor_Scalar)
class PowTensorScalar(Pow, FinalATenOp):
    """The final specialization of `torch.ops.aten.pow.Tensor_Scalar`.

    Raises a tensor to a scalar power.

    Attributes:
        this (Node): The first tensor operand
        other (Number): The second scalar operand
    """

    this: Node
    other: Number


@Pow.register(torch.ops.aten.pow.Tensor_Tensor)
class PowTensorTensor(Pow, FinalATenOp):
    """The final specialization of `torch.ops.aten.pow.Tensor_Tensor`.

    Raises a tensor to a tensor power.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
    """

    this: Node
    other: Node


@Pow.register(torch.ops.aten.pow.Scalar_Scalar)
class PowScalarScalar(Pow, FinalATenOp):
    """The final specialization of `torch.ops.aten.pow.Scalar_Scalar`.

    Raises a scalar to a scalar power.

    Attributes:
        this (Number): The first scalar operand
        other (Number): The second scalar operand
    """

    this: Number
    other: Number


class Sub(BinaryElementwise):
    """Binary elementwise subtraction operation.

    Attributes:
        this (Node | Number): The first operand
        other (Node | Number): The second operand
    """

    @property
    def is_commutative(self) -> Literal[False]:
        return False


@Sub.register(torch.ops.aten.sub.Tensor)
class SubTensor(Sub, FinalATenOp):
    """The final specialization of `torch.ops.aten.sub.Tensor`.

    Subtracts a tensor from a tensor or scalar.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
        asterisk (None): Placeholder indicating mandatory keyword argument
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this - alpha * other`. Defaults to 1.
    """

    other: Node
    asterisk: None = Asterisk
    alpha: Number = 1


class SubScalarTensor(SubTensor):
    """A further specialization of `torch.ops.aten.sub.Tensor`.

    Subtracts a scalar from a tensor.

    Attributes:
        this (Node): The first tensor operand
        other (Number): The second scalar operand
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this - alpha * other`. Defaults to 1.
    """

    this: Node


class SubTensorTensor(SubTensor):
    """A further specialization of `torch.ops.aten.sub.Tensor`.

    Subtracts a tensor from another tensor.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this - alpha * other`. Defaults to 1.
    """

    this: Node


@Sub.register(torch.ops.aten.sub.Scalar)
class SubScalar(Sub, FinalATenOp):
    """The final specialization of `torch.ops.aten.sub.Scalar`.

    Subtracts a scalar from a tensor or scalar.

    Attributes:
        this (Node | Number): The first operand
        other (Number): The second scalar operand
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this - alpha * other`. Defaults to 1.
    """

    other: Number
    alpha: Number = 1


class SubTensorScalar(SubScalar):
    """A further specialization of `torch.ops.aten.sub.Scalar`.

    Subtracts a scalar from a tensor.

    Attributes:
        this (Node): The first tensor operand
        other (Number): The second scalar operand
        alpha (Number, optional): Scaling factor for other. The operation performed for a non-default value is
            `this - alpha * other`. Defaults to 1.
    """

    this: Node


@Sub.register(torch.ops.aten.sub.default)
class SubDefault(Sub, FinalATenOp):
    """The final specialization of `torch.ops.aten.sub.default`.

    Subtracts a scalar from another scalar.

    Attributes:
        this (Number): The first scalar operand
        other (Number): The second scalar operand
    """

    this: Number
    other: Number


class Eq(BinaryElementwise):
    """Binary elementwise equality comparison operation.

    Attributes:
        this (Node | Number): The first operand
        other (Node | Number): The second operand
    """


@Eq.register(torch.ops.aten.eq.Tensor)
class EqTensor(Eq, FinalATenOp):
    """The final specialization of `torch.ops.aten.eq.Tensor`.

    Compares a tensor with another tensor elementwise for equality.

    Attributes:
        this (Node): The first tensor operand
        other (Node): The second tensor operand
    """

    this: Node
    other: Node

    @property
    def is_commutative(self) -> Literal[True]:
        return True


@Eq.register(torch.ops.aten.eq.Scalar)
class EqScalar(Eq, FinalATenOp):
    """The final specialization of `torch.ops.aten.eq.Scalar`.

    Compares a tensor with a scalar elementwise for equality.

    Attributes:
        this (Node): The first tensor operand
        other (Number): The second scalar operand
    """

    this: Node
    other: Number

    @property
    def is_commutative(self) -> Literal[False]:
        return False
