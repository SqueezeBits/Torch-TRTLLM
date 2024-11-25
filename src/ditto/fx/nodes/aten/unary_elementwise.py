# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch

from .unary import Unary


class UnaryElementwise(Unary):
    ...


@UnaryElementwise.final(torch.ops.aten.sqrt.default)
class Sqrt(UnaryElementwise):
    ...
