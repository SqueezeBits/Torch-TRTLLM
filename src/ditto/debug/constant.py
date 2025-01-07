from ctypes import c_char
from typing import overload

import onnx_graphsurgeon as gs
import torch
from onnx import TensorProto
from onnx.helper import make_tensor
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
    if tensor is None:
        assert shape is not None and dtype is not None
        tensor = torch.zeros(shape, dtype=dtype)
    else:
        tensor = tensor.cpu().contiguous()
    return gs.Constant(name=name, values=make_lazy_values(name, tensor))


def make_lazy_values(name: str, tensor: torch.Tensor) -> LazyValues:
    tensor_proto = TensorProto(
        name=name,
        dims=(*tensor.shape,),
        data_type=DataType(tensor.dtype).to(TensorProto.DataType),
    )
    return LazyValues(tensor_proto)
