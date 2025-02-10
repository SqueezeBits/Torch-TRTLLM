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

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Any, ClassVar

import torch
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter, field_serializer, field_validator, model_validator
from sympy import Integer
from torch._dynamo.source import SyntheticLocalSource
from torch.export import Dim
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, StrictMinMaxConstraint
from torch.utils._sympy.value_ranges import ValueRanges
from typing_extensions import Self

from ..contexts import detailed_sym_node_str
from ..types import ExportDim


class DynamicDimensionCacheConflictError(RuntimeError):
    """An error that occurs when something is conflicting with the cache of dynamic dimensions."""


class DynamicDimensionType(BaseModel, ABC):
    """A base class for dynamic dimensions.

    This class is used to create dynamic dimensions that can be used in the model.

    Attributes:
        CACHE (ClassVar[dict[str, Self]]): A class variable that caches the dynamic dimensions.
        _shape_env (ClassVar[ShapeEnv | None]): A class variable that stores the shape environment.
        _sym_int (torch.SymInt | None): A private attribute that stores the symbolic integer.
    """

    CACHE: ClassVar[dict[str, Self]] = {}
    _shape_env: ClassVar[ShapeEnv | None] = None
    _sym_int: torch.SymInt | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def cache_objects(self) -> Self:
        """Cache the dynamic dimension if it is not already cached.

        Returns:
            Self: The dynamic dimension

        Raises:
            DynamicDimensionCacheConflictError: If a dynamic dimension with the same name but
                with at least one different attributes is already cached.
        """
        if self.name in self.CACHE:
            if (cached_self := self.CACHE[self.name]).model_dump() != self.model_dump():
                raise DynamicDimensionCacheConflictError(
                    "Creating dynamic dimensions with the same name but with at least one different attributes is not "
                    f"allowed. Existing one was {repr(cached_self)} but tried to declare another one with the same "
                    f"name: {repr(self)}"
                )
            logger.trace(f"Using cached dynamic dimension {self.name}")
            return cached_self
        self.CACHE[self.name] = self
        return self

    @property
    def sym_int(self) -> torch.SymInt:
        """Get the symbolic integer of the dynamic dimension.

        Returns:
            torch.SymInt: The symbolic integer of the dynamic dimension
        """
        if self._sym_int is None:
            self._sym_int = self._create_sym_int()
        return self._sym_int

    @sym_int.setter
    def sym_int(self, other: torch.SymInt) -> None:
        """Set the symbolic integer of the dynamic dimension.

        Args:
            other (torch.SymInt): The symbolic integer to set

        Raises:
            RuntimeError: If the symbolic node is not a `SymNode` object or if the shape environment
                is not a `ShapeEnv` object
        """
        if isinstance(sym_node := other.node, SymNode) and isinstance(shape_env := sym_node.shape_env, ShapeEnv):
            if DynamicDimensionType._shape_env is not None and DynamicDimensionType._shape_env is not shape_env:
                logger.warning(
                    "Interleaving between two different shape environments has been detected! "
                    "You might experience failure in graph module fake tensor propagation!"
                )
            DynamicDimensionType._shape_env = shape_env
            self._sym_int = other
        else:
            with detailed_sym_node_str():
                raise RuntimeError(f"Failed to extract shape environment from `torch.SymInt` object {other}.")

    def _create_sym_int(self) -> torch.SymInt:
        """Create a new symbolic integer for the dynamic dimension.

        Returns:
            torch.SymInt: The symbolic integer of the dynamic dimension
        """
        if DynamicDimensionType._shape_env is None:
            logger.warning("Creating a new shape environment for dynamic shapes. This might cause unexpected behavior.")
            DynamicDimensionType._shape_env = ShapeEnv()
        shape_env = DynamicDimensionType._shape_env
        source = SyntheticLocalSource(local_name=f"Custom dynamic dimension {self.name}")
        expr = shape_env.create_symbol(
            self.example,
            source,
            dynamic_dim=DimDynamic.DYNAMIC,
            constraint_dim=StrictMinMaxConstraint(
                warn_only=True,
                vr=ValueRanges(Integer(self.min), Integer(self.max)),
            ),
        )
        sym_int = shape_env.create_symintnode(expr, hint=self.example)
        assert isinstance(sym_int, torch.SymInt)
        return sym_int

    @property
    @abstractmethod
    def export_dim(self) -> ExportDim | int:
        """Get the export dimension of the dynamic dimension.

        Returns:
            ExportDim | int: The export dimension of the dynamic dimension
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the dynamic dimension.

        Returns:
            str: The name of the dynamic dimension
        """

    @property
    @abstractmethod
    def min(self) -> int:
        """Get the minimum value of the dynamic dimension.

        Returns:
            int: The minimum value of the dynamic dimension
        """

    @property
    @abstractmethod
    def max(self) -> int:
        """Get the maximum value of the dynamic dimension.

        Returns:
            int: The maximum value of the dynamic dimension
        """

    @property
    @abstractmethod
    def opt(self) -> int:
        """Get the optimal value of the dynamic dimension.

        Returns:
            int: The optimal value of the dynamic dimension
        """

    @property
    @abstractmethod
    def example(self) -> int:
        """Get the example value of the dynamic dimension.

        Returns:
            int: The example value of the dynamic dimension
        """

    def _apply_binary_op(
        self,
        other: Self | int,
        op: Callable[[int, int], int],
        flip_order: bool = False,
    ) -> "DerivedDynamicDimension":
        """Apply a binary operation to the dynamic dimension.

        Args:
            other (Self | int): The other dynamic dimension or integer
            op (Callable[[int, int], int]): The binary operation to apply
            flip_order (bool): Whether to flip the order of the operands

        Returns:
            DerivedDynamicDimension: The derived dynamic dimension
        """
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
    """A dynamic dimension.

    Attributes:
        given_name (str): The name of the dynamic dimension
        given_min (int): The minimum value of the dynamic dimension
        given_opt (int): The optimal value of the dynamic dimension
        given_max (int): The maximum value of the dynamic dimension
        given_example (int | None): The example value of the dynamic dimension
    """

    given_name: str = Field(frozen=True, alias="name")
    given_min: int = Field(frozen=True, alias="min")
    given_opt: int = Field(frozen=True, alias="opt")
    given_max: int = Field(frozen=True, alias="max")
    given_example: int | None = Field(default=None, frozen=True, alias="example_for_export")

    @model_validator(mode="before")
    @classmethod
    def removeprefix(cls, data: Any) -> Any:
        """Remove the prefix from the given data.

        Args:
            data (Any): The data to remove the prefix from

        Returns:
            Any: The data with the prefix removed
        """
        if isinstance(data, dict):
            return {k.removeprefix("given_"): v for k, v in data.items()}
        return data

    @cached_property
    def export_dim(self) -> ExportDim | int:
        """Get the export dimension of the dynamic dimension.

        Returns:
            ExportDim | int: The export dimension of the dynamic dimension
        """
        return Dim(self.name, min=self.min, max=self.max) if self.min < self.max else self.min

    @property
    def name(self) -> str:
        """Get the name of the dynamic dimension.

        Returns:
            str: The name of the dynamic dimension
        """
        return self.given_name

    @property
    def min(self) -> int:
        """Get the minimum value of the dynamic dimension.

        Returns:
            int: The minimum value of the dynamic dimension
        """
        return self.given_min

    @property
    def max(self) -> int:
        """Get the maximum value of the dynamic dimension.

        Returns:
            int: The maximum value of the dynamic dimension
        """
        return self.given_max

    @property
    def opt(self) -> int:
        """Get the optimal value of the dynamic dimension.

        Returns:
            int: The optimal value of the dynamic dimension
        """
        return self.given_opt

    @cached_property
    def example(self) -> int:
        """Get the example value of the dynamic dimension.

        Returns:
            int: The example value of the dynamic dimension
        """
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
        """Validate that the dynamic dimension values meet requirements after model instantiation.

        Returns:
            Self: The validated dynamic dimension instance
        """
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
    """A derived dynamic dimension.

    Attributes:
        lhs (DynamicDimensionType | int): The left-hand side of the derived dynamic dimension
        rhs (DynamicDimensionType | int): The right-hand side of the derived dynamic dimension
        op (Callable[[int, int], int]): The operation to apply to the derived dynamic dimension
    """

    model_config = {"arbitrary_types_allowed": True}

    lhs: DynamicDimensionType | int = Field(frozen=True)
    rhs: DynamicDimensionType | int = Field(frozen=True)
    op: Callable[[int, int], int] = Field(frozen=True)

    @field_serializer("lhs", "rhs")
    def serialize_origins(self, origin: Any) -> dict[str, Any] | int:
        """Serialize the given argument.

        Args:
            origin (Any): The origin to serialize

        Returns:
            dict[str, Any] | int: The serialized origin
        """
        if isinstance(origin, DynamicDimensionType | int):
            return origin if isinstance(origin, int) else origin.model_dump()
        return origin

    @field_validator("lhs", "rhs", mode="before")
    @classmethod
    def validate_origins(cls, origin: Any) -> Any:
        """Validate the given argument.

        Args:
            origin (Any): The origin to validate

        Returns:
            Any: The validated origin
        """
        if isinstance(origin, int):
            return origin
        if isinstance(origin, dict):
            return TypeAdapter(DynamicDimension | DerivedDynamicDimension).validate_python(origin)
        return origin

    @cached_property
    def export_dim(self) -> ExportDim | int:
        """Get the export dimension of the derived dynamic dimension.

        Returns:
            ExportDim | int: The export dimension of the derived dynamic dimension
        """
        try:
            lhs = self.lhs.export_dim if isinstance(self.lhs, DynamicDimensionType) else self.lhs
            rhs = self.rhs.export_dim if isinstance(self.rhs, DynamicDimensionType) else self.rhs
            # pylint: disable-next=not-callable
            return self.op(lhs, rhs)
        except Exception as e:
            logger.warning(f"the derived dynamic dimension {self.name} will be detached (reason: {e})")
        return self.detach().export_dim

    @property
    def name(self) -> str:
        """Get the name of the derived dynamic dimension.

        Returns:
            str: The name of the derived dynamic dimension
        """
        lhs_name = self.lhs.name if isinstance(self.lhs, DynamicDimensionType) else f"{self.lhs}"
        rhs_name = self.rhs.name if isinstance(self.rhs, DynamicDimensionType) else f"{self.rhs}"
        op_name = {
            operator.add: "+",
            operator.floordiv: "//",
            operator.mul: "*",
            operator.sub: "-",
            operator.truediv: "/",
        }.get(self.op, f"_{self.op.__name__}_")
        return f"{lhs_name}{op_name}{rhs_name}"

    @cached_property
    def min(self) -> int:
        """Get the minimum value of the derived dynamic dimension.

        Returns:
            int: The minimum value of the derived dynamic dimension
        """
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
        """Get the maximum value of the derived dynamic dimension.

        Returns:
            int: The maximum value of the derived dynamic dimension
        """
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
        """Get the optimal value of the derived dynamic dimension.

        Returns:
            int: The optimal value of the derived dynamic dimension
        """
        lhs = self.lhs.opt if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        rhs = self.rhs.opt if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        # pylint: disable-next=not-callable
        return self.op(lhs, rhs)

    @property
    def example(self) -> int:
        """Get the example value of the derived dynamic dimension.

        Returns:
            int: The example value of the derived dynamic dimension
        """
        lhs = self.lhs.example if isinstance(self.lhs, DynamicDimensionType) else self.lhs
        rhs = self.rhs.example if isinstance(self.rhs, DynamicDimensionType) else self.rhs
        # pylint: disable-next=not-callable
        return self.op(lhs, rhs)

    def detach(self) -> DynamicDimension:
        """Detach the derived dynamic dimension.

        Returns:
            DynamicDimension: The detached dynamic dimension
        """
        return DynamicDimension(
            name=f"{self.name}_detached",
            min=self.min,
            opt=self.opt,
            max=self.max,
            example_for_export=self.example,
        )

    @model_validator(mode="after")
    def check_at_least_one_of_lhs_or_rhs_is_dynamic_dim(self) -> Self:
        """Validate a derived dynamic dimension after model instantiation.

        Returns:
            Self: The derived dynamic dimension
        """
        assert isinstance(self.lhs, DynamicDimensionType) or isinstance(
            self.rhs, DynamicDimensionType
        ), f"At least of lhs or rhs must be `DynamicDimension`, but got lhs={self.lhs}, rhs={self.rhs}"
        return self
