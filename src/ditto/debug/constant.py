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

from typing import overload

import onnx_graphsurgeon as gs
import torch
from onnx import TensorProto
from onnx_graphsurgeon.ir.tensor import LazyValues

from ..types import DataType


@overload
def make_constant(
    name: str,
    tensor: torch.Tensor,
) -> gs.Constant:
    ...


@overload
def make_constant(
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> gs.Constant:
    ...


def make_constant(
    name: str,
    tensor: torch.Tensor | None = None,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> gs.Constant:
    """Create a constant node from a tensor with shape and dtype.

    Args:
        name (str): The name of the constant node.
        tensor (torch.Tensor | None): The tensor to create the constant node from.
        shape (tuple[int, ...] | None): The shape of the constant node.
        dtype (torch.dtype | None): The dtype of the constant node.

    Returns:
        gs.Constant: The constant node.
    """
    if tensor is None:
        assert shape is not None and dtype is not None
        tensor = torch.zeros(shape, dtype=dtype)
    else:
        tensor = tensor.cpu().contiguous()
    return gs.Constant(name=name, values=make_lazy_values(name, tensor))


def make_lazy_values(name: str, tensor: torch.Tensor) -> LazyValues:
    """Create a lazy values object from a tensor.

    Args:
        name (str): The name of the tensor.
        tensor (torch.Tensor): The tensor to create the lazy values object from.

    Returns:
        LazyValues: The lazy values object.
    """
    tensor_proto = TensorProto(
        name=name,
        dims=(*tensor.shape,),
        data_type=DataType(tensor.dtype).to(TensorProto.DataType),
    )
    return LazyValues(tensor_proto)
