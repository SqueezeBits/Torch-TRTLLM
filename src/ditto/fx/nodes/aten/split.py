# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ....types import SymbolicInteger
from .aten_op import ATenOp, FinalATenOp


class Split(ATenOp):
    """Base class representing ATen split op overload packet.

    Attributes:
        this (Node): The input tensor node to split
        split_size (list[SymbolicInteger] | SymbolicInteger): Size of each split or list of sizes
        dim (int): Dimension along which to split the tensor, defaults to 0
    """

    this: Node
    split_size: list[SymbolicInteger] | SymbolicInteger
    dim: int = 0


@Split.register(torch.ops.aten.split.default)
class SplitDefault(Split, FinalATenOp):
    """ATen split operation that splits tensor into equal sized chunks.

    Attributes:
        this (Node): The input tensor node to split
        split_size (list[int]): Size of each split chunk
        dim (int): Dimension along which to split the tensor, defaults to 0
    """

    this: Node
    split_size: list[int]
    dim: int = 0


@Split.register(torch.ops.aten.split.sizes)
class SplitSizes(Split, FinalATenOp):
    """ATen split operation that splits tensor into chunks of specified sizes.

    Attributes:
        this (Node): The input tensor node to split
        split_size (list[torch.SymInt]): List of sizes for each split chunk
        dim (int): Dimension along which to split the tensor, defaults to 0
    """

    this: Node
    split_size: list[torch.SymInt]
    dim: int = 0


@Split.register(torch.ops.aten.split.Tensor)
class SplitTensor(Split, FinalATenOp):
    """ATen split operation that splits tensor using a size specified by another tensor.

    Attributes:
        this (Node): The input tensor node to split
        split_size (SymbolicInteger): Size of each split specified by a tensor
        dim (int): Dimension along which to split the tensor, defaults to 0
    """

    this: Node
    split_size: SymbolicInteger
    dim: int = 0
