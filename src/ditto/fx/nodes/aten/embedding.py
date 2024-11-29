# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import SymbolicInteger
from .aten_op import ATenOp


@ATenOp.final(torch.ops.aten.embedding.default)
class Embedding(ATenOp):
    weight: Node
    indices: Node
    padding_idx: SymbolicInteger = -1
    scale_grad_by_freq: bool = False
    sparse: bool = False
