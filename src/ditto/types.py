# pylint: disable=unused-import, invalid-name
from collections.abc import Callable
from functools import cache
from typing import ClassVar, TypeVar, cast

import tensorrt as trt
import torch
from onnx import TensorProto
from pydantic import BaseModel, ConfigDict
from torch._C import _SDPBackend as SDPBackend  # noqa: F401
from torch.export.dynamic_shapes import _Dim as ExportDim  # noqa: F401
from torch.fx import Node

BuiltInConstant = int | float | bool | None
DeviceLikeType = str | torch.device | int
Number = int | float | bool
SymbolicInteger = int | torch.SymInt
SymbolicShape = tuple[SymbolicInteger, ...]
ShapeArg = list[int | Node]


class StrictlyTyped(BaseModel):
    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "validate_default": True,
    }


class UnsupportedDataTypeConversionError(TypeError):
    """Error indicating unsupported data type conversion between two different frameworks."""


FromDataType = TypeVar("FromDataType", torch.dtype, trt.DataType, TensorProto.DataType, int)
OtherDataType = TypeVar("OtherDataType", torch.dtype, trt.DataType, TensorProto.DataType, int)
ToDataType = TypeVar("ToDataType", torch.dtype, trt.DataType, TensorProto.DataType, int)


class DataType:
    MAPPING: ClassVar[dict[tuple[type, type], dict]] = {}

    def __init__(self, dtype: FromDataType) -> None:
        self.dtype: torch.dtype | trt.DataType | int = dtype

    def to(self, data_type: type[ToDataType]) -> ToDataType:
        if data_type is TensorProto.DataType:  # TensorProto.DataType is not actually a type
            if isinstance(self.dtype, int):
                return cast(TensorProto.DataType, self.dtype)  # type: ignore[return-value]
        elif isinstance(self.dtype, data_type):
            return self.dtype
        actual_type: type = int if data_type is TensorProto.DataType else data_type  # type: ignore[assignment]
        assert (
            type_pair := (type(self.dtype), actual_type)
        ) in self.MAPPING, f"Conversion from {type_pair[0]} to {type_pair[1]} is not defined"
        try:
            return self.MAPPING[type_pair][self.dtype]
        except KeyError as e:
            raise UnsupportedDataTypeConversionError(f"{self.dtype} cannot be converted to {data_type}") from e

    @classmethod
    def define_from(
        cls,
        dtype_mapping: Callable[[], dict[FromDataType, ToDataType]],
    ) -> Callable[[], dict[FromDataType, ToDataType]]:
        m = dtype_mapping()
        _from, _to = next(iter(m.items()))
        type_from = type(_from)
        type_to = type(_to)
        cls._register((type_from, type_to), m)
        cls._register((type_to, type_from), {v: k for k, v in m.items()})
        return cache(dtype_mapping)

    @classmethod
    def define_by_composition(
        cls,
        m0: dict[FromDataType, OtherDataType],
        m1: dict[OtherDataType, ToDataType],
    ) -> None:
        m = {k: m1[v] for k, v in m0.items() if v in m1}
        _from = next(iter(m0.keys()))
        _to = next(iter(m1.values()))
        cls._register((type(_from), type(_to)), m)

    @classmethod
    def _register(
        cls,
        type_pair: tuple[type[FromDataType], type[ToDataType]],
        mapping: dict[FromDataType, ToDataType],
    ) -> None:
        assert type_pair not in cls.MAPPING, (
            f"{cls.__name__}[{type_pair[0]}, {type_pair[1]}] is already defined with "
            f"mapping {cls.MAPPING[type_pair]}"
        )
        cls.MAPPING[type_pair] = mapping


@DataType.define_from
def trt_to_torch_dtype_mapping() -> dict[trt.DataType, torch.dtype]:
    """Create `trt.DataType` to `torch.dtype` compatibility map.

    * All TensorRT data types can be mapped to a PyTorch data type.

    Returns:
        dict[trt.DataType, torch.dtype]: the compatibility map.
    """
    return {
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.FP8: torch.float8_e4m3fn,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT8: torch.int8,
        trt.DataType.UINT8: torch.uint8,
    }


@DataType.define_from
def torch_to_onnx_dtype_mapping() -> dict[torch.dtype, TensorProto.DataType]:
    """Create `torch.dtype` to `onnx.TensorProto.DataType` compatibility map.

    * Some of ONNX data types cannot be converted to PyTorch data types.

    Returns:
        dict[torch.dtype, onnx.TensorProto.DataType]: the compatibility map.
    """
    return {
        torch.float32: TensorProto.FLOAT,
        torch.uint8: TensorProto.UINT8,
        torch.int8: TensorProto.INT8,
        torch.uint16: TensorProto.UINT16,
        torch.int16: TensorProto.INT16,
        torch.int32: TensorProto.INT32,
        torch.int64: TensorProto.INT64,
        torch.bool: TensorProto.BOOL,
        torch.float16: TensorProto.FLOAT16,
        torch.double: TensorProto.DOUBLE,
        torch.uint32: TensorProto.UINT32,
        torch.uint64: TensorProto.UINT64,
        torch.complex64: TensorProto.COMPLEX64,
        torch.complex128: TensorProto.COMPLEX128,
        torch.bfloat16: TensorProto.BFLOAT16,
        torch.float8_e4m3fn: TensorProto.FLOAT8E4M3FN,
        torch.float8_e4m3fnuz: TensorProto.FLOAT8E4M3FNUZ,
        torch.float8_e5m2: TensorProto.FLOAT8E5M2,
        torch.float8_e5m2fnuz: TensorProto.FLOAT8E5M2FNUZ,
    }


# All PyTorch data types converted from TensorRT data types can be mapped to ONNX data types.
DataType.define_by_composition(trt_to_torch_dtype_mapping(), torch_to_onnx_dtype_mapping())
