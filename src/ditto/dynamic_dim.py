import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property

from loguru import logger
from pydantic import BaseModel, Field, model_validator
from torch.export import Dim
from typing_extensions import Self

from .types import DimType


class DynamicDimensionType(BaseModel, ABC):
    @property
    @abstractmethod
    def export_dim(self) -> DimType | int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def min(self) -> int:
        ...

    @property
    @abstractmethod
    def max(self) -> int:
        ...

    @property
    @abstractmethod
    def opt(self) -> int:
        ...

    @property
    @abstractmethod
    def example(self) -> int:
        ...

    def _apply_binary_op(
        self,
        other: Self | int,
        op: Callable[[int, int], int],
        flip_order: bool = False,
    ) -> "DerivedDynamicDimension":
        if isinstance(other, DynamicDimensionType | int):
            return DerivedDynamicDimension(
                lhs=other if flip_order else self,
                rhs=self if flip_order else other,
                op=op,
            )
        raise ValueError(f"Cannot {op.__name__} {self} and {other}")

    def __add__(self, other: Self | int) -> "DerivedDynamicDimension":
        return self._apply_binary_op(other, operator.add)

    def __sub__(self, other: Self | int) -> "DerivedDynamicDimension":
        return self._apply_binary_op(other, operator.sub)

    def __mul__(self, other: Self | int) -> "DerivedDynamicDimension":
        return self._apply_binary_op(other, operator.mul)

    def __rmul__(self, other: Self | int) -> "DerivedDynamicDimension":
        return self._apply_binary_op(other, operator.mul, flip_order=True)

    def __floordiv__(self, other: Self | int) -> "DerivedDynamicDimension":
        return self._apply_binary_op(other, operator.floordiv)


class DynamicDimension(DynamicDimensionType):
    given_name: str = Field(frozen=True, alias="name")
    given_min: int = Field(frozen=True, alias="min")
    given_opt: int = Field(frozen=True, alias="opt")
    given_max: int = Field(frozen=True, alias="max")
    given_example: int | None = Field(default=None, frozen=True, alias="example_for_export")

    @cached_property
    def export_dim(self) -> DimType | int:
        return Dim(self.name, min=self.min, max=self.max) if self.min < self.max else self.min

    @property
    def name(self) -> str:
        return self.given_name

    @property
    def min(self) -> int:
        return self.given_min

    @property
    def max(self) -> int:
        return self.given_max

    @property
    def opt(self) -> int:
        return self.given_opt

    @cached_property
    def example(self) -> int:
        ex = min(max(self.opt, 2), self.max) if self.given_example is None else self.given_example
        if ex < 2 and self.min < self.max:
            the_example_size = (
                "the inferred example size" if self.given_example is None else "the provided example size"
            )
            logger.warning(
                f"{the_example_size} {ex} for `torch.export` of the dimension {self.name} is less than 2. "
                "This can be problematic for `torch.export` as it often considers the dimension of size 1 as static. "
                f"Please set the `example_for_export` for {self.name} greater than or equal to 2"
            )
        return ex

    @model_validator(mode="after")
    def check_given_values(self) -> Self:
        assert self.name.isidentifier(), f"name must be an identifier but got '{self.name}'"
        assert (
            0 <= self.min <= self.max
        ), f"0 <= min < max must be satified, but {self.name} got min={self.min}, max={self.max}"
        assert (
            self.min <= self.opt <= self.max
        ), f"min <= opt <= max must be satisfied, but {self.name} got min={self.min}, opt={self.opt}, max={self.max}"
        assert self.min <= self.example <= self.max, (
            "min <= example <= max must be satisfied, "
            f"but {self.name} got min={self.min}, opt={self.example}, max={self.max}"
        )
        return self


class DerivedDynamicDimension(DynamicDimensionType):
    model_config = {"arbitrary_types_allowed": True}

    lhs: DynamicDimensionType | int = Field(frozen=True)
    rhs: DynamicDimensionType | int = Field(frozen=True)
    op: Callable[[int, int], int] = Field(frozen=True)

    @cached_property
    def export_dim(self) -> DimType | int:
        try:
            lhs = self.lhs.export_dim if isinstance(self.lhs, DynamicDimensionType) else self.lhs
            rhs = self.rhs.export_dim if isinstance(self.rhs, DynamicDimensionType) else self.rhs
            # pylint: disable-next=not-callable
            return self.op(lhs, rhs)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"the derived dynamic dimension {self.name} will be detached (reason: {e})")
        return self.detach().export_dim

    @property
    def name(self) -> str:
        lhs_name = self.lhs.name if isinstance(self.lhs, DynamicDimensionType) else f"{self.lhs}"
        rhs_name = self.rhs.name if isinstance(self.rhs, DynamicDimensionType) else f"{self.rhs}"
        return f"{lhs_name}_{self.op.__name__}_{rhs_name}"

    @cached_property
    def min(self) -> int:
        lhs_min = self.lhs.min if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        lhs_max = self.lhs.max if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        rhs_min = self.rhs.min if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        rhs_max = self.rhs.max if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        return min(
            (
                # pylint: disable-next=not-callable
                self.op(lhs, rhs)
                for lhs, rhs in (
                    (lhs_min, rhs_min),
                    (lhs_min, rhs_max),
                    (lhs_max, rhs_min),
                    (lhs_max, rhs_max),
                )
            )
        )

    @cached_property
    def max(self) -> int:
        lhs_min = self.lhs.min if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        lhs_max = self.lhs.max if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        rhs_min = self.rhs.min if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        rhs_max = self.rhs.max if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        return max(
            (
                # pylint: disable-next=not-callable
                self.op(lhs, rhs)
                for lhs, rhs in (
                    (lhs_min, rhs_min),
                    (lhs_min, rhs_max),
                    (lhs_max, rhs_min),
                    (lhs_max, rhs_max),
                )
            )
        )

    @cached_property
    def opt(self) -> int:
        lhs = self.lhs.opt if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        rhs = self.rhs.opt if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        # pylint: disable-next=not-callable
        return self.op(lhs, rhs)

    @property
    def example(self) -> int:
        lhs = self.lhs.example if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        rhs = self.rhs.example if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        # pylint: disable-next=not-callable
        return self.op(lhs, rhs)

    def detach(self) -> DynamicDimension:
        return DynamicDimension(
            name=f"{self.name}_detached",
            min=self.min,
            opt=self.opt,
            max=self.max,
            example_for_export=self.example,
        )

    @model_validator(mode="after")
    def check_at_least_one_of_lhs_or_rhs_is_dynamic_dim(self) -> Self:
        assert isinstance(self.lhs, DynamicDimensionType) or isinstance(
            self.rhs, DynamicDimensionType
        ), f"At least of lhs or rhs must be `DynamicDimension`, but got lhs={self.lhs}, rhs={self.rhs}"
        return self
