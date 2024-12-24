# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import Literal

import torch
from torch.fx.node import Node

from .aten_op import FinalATenOp
from .binary import Binary


@Binary.register(torch.ops.aten.mm.default)
class MM(Binary, FinalATenOp):
    this: Node
    other: Node

    @property
    def is_commutative(self) -> Literal[False]:
        return False


@Binary.register(torch.ops.aten.bmm.default)
class BMM(Binary, FinalATenOp):
    this: Node
    other: Node

    @property
    def is_commutative(self) -> Literal[False]:
        return False
