# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import SymbolicInteger
from .aten_op import ATenOp


@ATenOp.final(torch.ops.aten.split.default, torch.ops.aten.split.sizes)
class Split(ATenOp):
    this: Node
    split_size: list[SymbolicInteger] | SymbolicInteger
    dim: int = 0
