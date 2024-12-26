# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
import torch

from .aten_op import FinalATenOp
from .unary_elementwise import UnaryElementwise


class Activation(UnaryElementwise):
    ...


@Activation.register(torch.ops.aten.elu.default)
class Elu(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.gelu.default)
class Gelu(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.hardsigmoid.default)
class HardSigmoid(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.leaky_relu.default)
class LeakyRelu(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.relu.default)
class Relu(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.sigmoid.default)
class Sigmoid(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.silu.default)
class Silu(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.softplus.default)
class Softplus(Activation, FinalATenOp):
    ...


@Activation.register(torch.ops.aten.tanh.default)
class Tanh(Activation, FinalATenOp):
    ...
