# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import Literal

import torch
from torch.fx.node import Node

from .binary import Binary


@Binary.final(torch.ops.aten.mm.default)
class MM(Binary):
    this: Node
    other: Node

    @property
    def is_commutative(self) -> Literal[False]:
        return False
