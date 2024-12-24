# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from .aten_op import ATenOp, FinalATenOp


class Softmax(ATenOp):
    this: Node
    dim: int


@Softmax.register(torch.ops.aten._softmax.default)
class SoftmaxDefault(Softmax, FinalATenOp):
    half_to_float: bool


@Softmax.register(torch.ops.aten._safe_softmax.default)
class SafeSoftmax(Softmax, FinalATenOp):
    dtype: torch.dtype | None = None
