from typing import Any

import torch
from pydantic import TypeAdapter, field_serializer, field_validator
from torch_tensorrt import Input

from ..types import ExportDim, StrictlyTyped
from .dynamic_dim import DerivedDynamicDimension, DynamicDimension, DynamicDimensionType


class TensorTypeHint(StrictlyTyped):
    shape: tuple[int | DynamicDimensionType, ...]
    dtype: torch.dtype

    @property
    def export_shape(self) -> tuple[int | ExportDim, ...]:
        return tuple(s.export_dim if isinstance(s, DynamicDimensionType) else s for s in self.shape)

    def as_spec(self, name: str) -> Input:
        if self.shape == (int_shape := tuple(s for s in self.shape if isinstance(s, int))):
            return Input.from_tensor(torch.zeros(int_shape, dtype=self.dtype))
        min_shape = tuple(s if isinstance(s, int) else s.min for s in self.shape)
        min_shape = tuple(max(1, s) for s in min_shape)
        opt_shape = tuple(s if isinstance(s, int) else s.opt for s in self.shape)
        max_shape = tuple(s if isinstance(s, int) else s.max for s in self.shape)
        example_shape = tuple(s if isinstance(s, int) else s.example for s in self.shape)
        return Input(
            name=name,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            dtype=self.dtype,
            format=torch.contiguous_format,
            torch_tensor=torch.zeros(example_shape, dtype=self.dtype),
        )

    @field_serializer("shape")
    def serialize_shape(self, shape: tuple[int | DynamicDimensionType, ...]) -> tuple[int | dict[str, Any], ...]:
        return tuple(s.model_dump() if isinstance(s, DynamicDimensionType) else s for s in shape)

    @field_validator("shape", mode="before")
    @classmethod
    def validate_shape(cls, shape: Any) -> tuple[int | DynamicDimensionType, ...]:
        if isinstance(shape, tuple):
            return tuple(
                s if isinstance(s, int) else TypeAdapter(DynamicDimension | DerivedDynamicDimension).validate_python(s)
                for s in shape
            )
        return shape
