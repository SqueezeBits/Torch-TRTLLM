from functools import cached_property

from pydantic import BaseModel, Field, model_validator
from torch.export import Dim
from typing_extensions import Self

from .types import DimType


class DynamicDim(BaseModel):
    name: str = Field(frozen=True)
    min: int = Field(frozen=True)
    opt: int = Field(frozen=True)
    max: int = Field(frozen=True)

    @cached_property
    def export_dim(self) -> DimType:
        return Dim(self.name, min=self.min, max=self.max)

    @model_validator(mode="after")
    def check_sizes(self) -> Self:
        assert self.min <= self.opt <= self.max
        return self
