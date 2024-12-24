# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import SymbolicInteger
from .aten_op import ATenOp, FinalATenOp


class Split(ATenOp):
    this: Node
    split_size: list[SymbolicInteger] | SymbolicInteger
    dim: int = 0


@Split.register(torch.ops.aten.split.default)
class SplitDefault(Split, FinalATenOp):
    this: Node
    split_size: SymbolicInteger
    dim: int = 0


@Split.register(torch.ops.aten.split.sizes)
class SplitSizes(Split, FinalATenOp):
    this: Node
    split_size: list[SymbolicInteger]
    dim: int = 0
