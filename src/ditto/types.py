# pylint: disable=unused-import
from typing import ClassVar

from onnx import TensorProto
import tensorrt as trt
import torch
import torch_tensorrt
from pydantic import BaseModel, ConfigDict
from torch._C import _SDPBackend as SDPBackend  # noqa: F401
from torch.export.dynamic_shapes import _Dim as ExportDim  # noqa: F401

BuiltInConstant = int | float | bool | None
DeviceLikeType = str | torch.device | int
Number = int | float | bool
SymbolicInteger = int | torch.SymInt
SymbolicShape = tuple[SymbolicInteger, ...]


class StrictlyTyped(BaseModel):
    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "validate_default": True,
    }


def map_torch_to_onnx_dtype(torch_type: torch.dtype) -> TensorProto.DataType:
    return {
        torch.bfloat16: TensorProto.BFLOAT16,
        torch.bool: TensorProto.BOOL,
        torch.float16: TensorProto.FLOAT16,
        torch.float32: TensorProto.FLOAT,
        torch.int32: TensorProto.INT32,
        torch.int64: TensorProto.INT64,
    }[torch_type]


def map_torch_to_trt_dtype(torch_type: torch.dtype) -> trt.DataType:
    return torch_tensorrt._from(torch_type).to(trt.DataType)


def map_trt_to_onnx_dtype(trt_type: trt.DataType) -> TensorProto.DataType:
    return {
        trt.DataType.BF16: TensorProto.BFLOAT16,
        trt.DataType.BOOL: TensorProto.BOOL,
        trt.DataType.HALF: TensorProto.FLOAT16,
        trt.DataType.FLOAT: TensorProto.FLOAT,
        trt.DataType.INT32: TensorProto.INT32,
        trt.DataType.INT64: TensorProto.INT64,
    }[trt_type]


def map_trt_to_torch_dtype(trt_type: trt.DataType) -> torch.dtype:
    return torch_tensorrt._from(trt_type).to(torch.dtype)