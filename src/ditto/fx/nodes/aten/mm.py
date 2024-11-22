# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from .binary import Binary


@Binary.final(torch.ops.aten.mm.default)
class MM(Binary):
    this: Node
    other: Node
