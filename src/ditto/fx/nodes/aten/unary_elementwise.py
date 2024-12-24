# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch

from .aten_op import FinalATenOp
from .unary import Unary


class UnaryElementwise(Unary):
    ...


@UnaryElementwise.register(torch.ops.aten.sqrt.default)
class Sqrt(UnaryElementwise, FinalATenOp):
    ...
