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

import warnings
from typing import Generic, Literal, TypeVar, get_args

import tensorrt as trt
from pydantic import ConfigDict
from pydantic.warnings import GenericBeforeBaseModelWarning
from torch_tensorrt import dtype
from typing_extensions import Self, Unpack

from ...types import StrictlyTyped

# pylint: disable-next=invalid-name
TRTEnumFlag = TypeVar(
    "TRTEnumFlag",
    trt.AllocatorFlag,
    trt.BuilderFlag,
    trt.NetworkDefinitionCreationFlag,
    trt.OnnxParserFlag,
    trt.QuantizationFlag,
    trt.SerializationFlag,
    trt.TempfileControlFlag,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=GenericBeforeBaseModelWarning)

    class BitmaskFlags(Generic[TRTEnumFlag], StrictlyTyped):
        def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
            super().__init_subclass__(**kwargs)
            for name, field in cls.model_fields.items():
                assert field.annotation is bool, (
                    f"A subclass of {BitmaskFlags.__name__} must have bool type fields, but {cls.__name__} has "
                    f"invalid field '{name}' of type {field.annotation}"
                )

        @classmethod
        def from_bitmask(cls, bitmask: int) -> Self:
            # pylint: disable-next=no-member
            type_arg = get_args(cls.__orig_bases__[0])[0]  # type: ignore[attr-defined]  # noqa: N806
            return cls.model_validate(
                {name.lower(): ((bitmask >> flag.value) & 1) for name, flag in type_arg.__members__.items()}
            )

        @property
        def bitmask(self) -> int:
            flags = 0
            data = self.model_dump()
            # pylint: disable-next=no-member
            type_arg = get_args(type(self).__orig_bases__[0])[0]  # type: ignore[attr-defined]  # noqa: N806
            for name, flag in type_arg.__members__.items():
                assert (key := name.lower()) in data, f"No such {key=} in {data=}"
                if data[name.lower()]:
                    flags |= 1 << flag.value
            return flags

    class TensorRTBuilderFlags(BitmaskFlags[trt.BuilderFlag]):
        fp16: bool = False
        bf16: bool = False
        int8: bool = False
        debug: bool = False
        gpu_fallback: bool = False
        refit: bool = False
        disable_timing_cache: bool = False
        tf32: bool = True
        sparse_weights: bool = False
        safety_scope: bool = False
        obey_precision_constraints: bool = False
        prefer_precision_constraints: bool = False
        direct_io: bool = False
        reject_empty_algorithms: bool = False
        version_compatible: bool = False
        exclude_lean_runtime: bool = False
        fp8: bool = False
        error_on_timing_cache_miss: bool = False
        disable_compilation_cache: bool = False
        weightless: bool = False
        strip_plan: bool = False
        refit_identical: bool = False
        weight_streaming: bool = False
        int4: bool = False
        refit_individual: bool = False
        strict_nans: bool = False
        monitor_memory: bool = False

        @property
        def enabled_precisions(self) -> set[dtype]:
            return {
                t
                for t, enabled in {
                    dtype.fp16: self.fp16,
                    dtype.int8: self.int8,
                    dtype.fp8: self.fp8,
                    dtype.bf16: self.bf16,
                }.items()
                if enabled
            }

    class TensorRTQuantizationFlags(BitmaskFlags[trt.QuantizationFlag]):
        calibrate_before_fusion: bool = False

    class TensorRTNetworkCreationFlags(BitmaskFlags[trt.NetworkDefinitionCreationFlag]):
        explicit_batch: Literal[True] = True
        strongly_typed: bool = True
