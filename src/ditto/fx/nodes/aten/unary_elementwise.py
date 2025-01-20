# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch

from .aten_op import FinalATenOp
from .unary import Unary


class UnaryElementwise(Unary):
    """Base class for unary elementwise operations."""


@UnaryElementwise.register(torch.ops.aten.neg.default)
class Neg(UnaryElementwise, FinalATenOp):
    """Elementwise negation operation."""


@UnaryElementwise.register(torch.ops.aten.sqrt.default)
class Sqrt(UnaryElementwise, FinalATenOp):
    """Elementwise square root operation."""
