# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import Number
from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.addmm.default)
class AddMM(FinalATenOp):
    bias: Node
    mat1: Node
    mat2: Node
    beta: Number = 1
    alpha: Number = 1
