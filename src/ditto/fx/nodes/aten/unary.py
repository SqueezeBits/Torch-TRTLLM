# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from torch.fx.node import Node

from .aten_op import ATenOp


class Unary(ATenOp):
    this: Node
