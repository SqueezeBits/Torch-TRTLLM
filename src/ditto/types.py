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

# mypy: disable-error-code="misc"
# pylint: disable=unused-import, invalid-name

from collections.abc import Callable
from functools import cache
from inspect import isclass
from typing import Any, ClassVar, TypeVar, cast, overload

import tensorrt as trt
import torch
from onnx import TensorProto
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from torch._C import _SDPBackend as SDPBackend  # noqa: F401
from torch.export.dynamic_shapes import _Dim as ExportDim  # noqa: F401
from torch.fx import Node
from typing_extensions import Unpack

BuiltInConstant = int | float | bool | None
DeviceLikeType = str | torch.device | int
NodeCriterion = Callable[[Node], bool]
Number = int | float | bool
SymbolicInteger = torch.SymInt | int
SymbolicShape = tuple[SymbolicInteger, ...]  # type: ignore[valid-type]
ShapeArg = list[int | Node]

T = TypeVar("T")


def verify(value: Any, *, as_type: type[T], coerce: bool = False, **config: Unpack[ConfigDict]) -> T | None:
    """Verify that the value is of the specified type.

    Args:
        value (Any): The value to verify
        as_type (type[T]): The target type to verify the value against
        coerce (bool): Whether to coerce the value to the target type. Defaults to False.
        **config (Unpack[ConfigDict]): The configuration to use for type verification.
            Ignored when `as_type` is a `pydantic.BaseModel` subclass.

    Returns:
        T | None: The (coerced) value if it is of the specified type, None otherwise
    """
    try:
        coerced_value = (
            as_type.model_validate(value)
            if (isclass(as_type) and not hasattr(as_type, "__origin__") and issubclass(as_type, BaseModel))
            else TypeAdapter(as_type, config={"arbitrary_types_allowed": True, **config}).validate_python(value)
        )
        return coerced_value if coerce else value
    except ValidationError:
        return None


class StrictlyTyped(BaseModel):
    """A base class for strictly typed models with enhanced validation.

    This class extends `BaseModel` and enforces strict type checking and validation
    rules for its fields. It is configured to allow arbitrary types, validate field
    assignments, and validate default values.
    """

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "validate_default": True,
    }


class UnsupportedDataTypeConversionError(TypeError):
    """Error indicating unsupported data type conversion between two different frameworks."""


FromDataType = TypeVar("FromDataType", torch.dtype, trt.DataType, int)
OtherDataType = TypeVar("OtherDataType", torch.dtype, trt.DataType, int)
ToDataType = TypeVar("ToDataType", torch.dtype, trt.DataType, int)


class DataType:
    """A utility class for managing and converting between data types.

    This class facilitates conversions between various data types, such as PyTorch
    `torch.dtype`, TensorRT `trt.DataType`, and `onnx.TensorProto.DataType` (`int`). It uses
    a mapping mechanism to define conversion rules and supports both direct and
    composed mappings.

    Attributes:
        MAPPING (ClassVar[dict[tuple[type, type], dict]]): A class-level dictionary that
            stores mappings between data type pairs and their conversion rules
    """

    MAPPING: ClassVar[dict[tuple[type, type], dict]] = {}

    def __init__(self, dtype: FromDataType) -> None:
        self.dtype: torch.dtype | trt.DataType | int = dtype

    def to(self, data_type: type[ToDataType]) -> ToDataType:
        """Convert the stored data type to a specified target type.

        Args:
            data_type (type[ToDataType]): The target data type class.

        Returns:
            ToDataType: The converted data type.

        Raises:
            AssertionError: If a mapping for the conversion is not defined.
            UnsupportedDataTypeConversionError: If the conversion fails due to a missing
                specific mapping for the given value.
        """
        if data_type is TensorProto.DataType:  # TensorProto.DataType is not actually a type
            if isinstance(self.dtype, int):
                return cast(TensorProto.DataType, self.dtype)
        elif isinstance(self.dtype, data_type):
            return self.dtype
        actual_type: type = int if data_type is TensorProto.DataType else data_type
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
        """Register a direct mapping between two types.

        Args:
            dtype_mapping (Callable[[], dict[FromDataType, ToDataType]]): A callable that
                generates a mapping dictionary between the source and target types.

        Returns:
            Callable[[], dict[FromDataType, ToDataType]]: The cached callable that
                provides the mapping.
        """
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
        """Define a mapping by composing two existing mappings.

        Args:
            m0 (dict[FromDataType, OtherDataType]): The first mapping, which connects
                the source type to an intermediate type.
            m1 (dict[OtherDataType, ToDataType]): The second mapping, which connects the
                intermediate type to the target type.

        Returns:
            None
        """
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
        """Register a specific mapping between two types.

        Args:
            type_pair (tuple[type[FromDataType], type[ToDataType]]): A tuple containing
                the source and target types.
            mapping (dict[FromDataType, ToDataType]): A dictionary defining the
                conversion rules between the source and target types.

        Raises:
            AssertionError: If a mapping for the given type pair is already defined.
        """
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
def torch_to_onnx_dtype_mapping() -> dict[torch.dtype, int]:
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


ExpectedType = TypeVar("ExpectedType")


@overload
def expect_identical(value: ExpectedType, *others: ExpectedType) -> ExpectedType | None:
    ...


@overload
def expect_identical(value: Any, *others: Any, expecting_type: type[ExpectedType]) -> ExpectedType | None:
    ...


def expect_identical(value: Any, *others: Any, expecting_type: type[ExpectedType] | None = None) -> ExpectedType | None:
    """Compare multiple values for equality and optionally check their type.

    Args:
        value: First value to compare
        *others: Additional values to compare against the first value
        expecting_type: Optional type to validate all values against. If None, no type checking is performed.

    Returns:
        The first value if all values are equal and match the specified type (if provided).
        None if any values are not equal or don't match the type.
    """
    if expecting_type is not None and not all(verify(v, as_type=expecting_type) is not None for v in (value, *others)):
        return None
    if all(v == value for v in others):
        return value
    return None
