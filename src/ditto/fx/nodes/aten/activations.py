# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
import torch

from .unary_elementwise import UnaryElementwise


class Activation(UnaryElementwise):
    ...


@Activation.final(torch.ops.aten.elu.default)
class Elu(Activation):
    ...


@Activation.final(torch.ops.aten.gelu.default)
class Gelu(Activation):
    ...


@Activation.final(torch.ops.aten.hardsigmoid.default)
class HardSigmoid(Activation):
    ...


@Activation.final(torch.ops.aten.leaky_relu.default)
class LeakyRelu(Activation):
    ...


@Activation.final(torch.ops.aten.relu.default)
class Relu(Activation):
    ...


@Activation.final(torch.ops.aten.sigmoid.default)
class Sigmoid(Activation):
    ...


@Activation.final(torch.ops.aten.softplus.default)
class Softplus(Activation):
    ...


@Activation.final(torch.ops.aten.tanh.default)
class Tanh(Activation):
    ...
