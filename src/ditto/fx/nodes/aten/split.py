# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


@Split.register(torch.ops.aten.split_with_sizes.default)
class SplitWithSizes(Split, FinalATenOp):
    """ATen split operation that splits tensor into chunks of specified sizes.

    Attributes:
        this (Node): The input tensor node to split
        split_size (list[int | torch.SymInt]): List of sizes for each split chunk
        dim (int): Dimension along which to split the tensor, defaults to 0
    """

    this: Node
    split_size: list[int | torch.SymInt]
    dim: int = 0
