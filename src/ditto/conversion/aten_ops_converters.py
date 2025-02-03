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

# pylint: disable=unused-argument
# mypy: disallow-untyped-decorators=False
from collections.abc import Sequence

import numpy as np
import tensorrt as trt
import torch
from loguru import logger
from pydantic import ValidationError, model_validator
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo import SourceIR
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    ConverterPriority,
    dynamo_tensorrt_converter,
    impl,
)
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    args_bounds_check,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    enforce_tensor_types,
    get_trt_tensor,
    set_layer_name,
)
from typing_extensions import Self

from ..fx.nodes.aten.utils import make_axis_nonnegative
from ..types import StrictlyTyped


@dynamo_tensorrt_converter(torch.ops.aten.all.default, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.all.dim, supports_dynamic_shapes=True)
@dynamo_tensorrt_converter(torch.ops.aten.all.dims, supports_dynamic_shapes=True)
def aten_ops_all(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert PyTorch's aten.all operation to TensorRT.

    Args:
        ctx (ConversionContext): Conversion context
        target (Target): Target operation
        args (tuple[Argument, ...]): Positional arguments
        kwargs (dict[str, Argument]): Keyword arguments
        name (str): Layer name

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Converted TensorRT tensor or sequence of tensors
    """
    return reduce_all(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        input_val=args[0],
        dim=args_bounds_check(args, 1, replacement=None),
        keepdim=args_bounds_check(args, 2, replacement=False),
    )


def reduce_all(
    ctx: ConversionContext,
    target: Target,
    source_ir: SourceIR | None,
    name: str,
    *,
    input_val: trt.ITensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> trt.ITensor:
    """Reduce tensor along specified dimensions using logical AND.

    Args:
        ctx (ConversionContext): Conversion context
        target (Target): Target operation
        source_ir (SourceIR | None): Source IR type
        name (str): Layer name
        input_val (trt.ITensor): Input tensor
        dim (int | Sequence[int] | None, optional): Dimensions to reduce along. Defaults to None.
        keepdim (bool, optional): Whether to keep reduced dimensions. Defaults to False.

    Returns:
        trt.ITensor: Reduced tensor
    """
    if (isinstance(input_val, trt.ITensor)) and (input_val.dtype == trt.bool):
        input_val = cast_trt_tensor(ctx, input_val, trt.int32, f"{name}_cast")

    abs_out = impl.unary.abs(
        ctx,
        target,
        source_ir,
        f"{name}_abs",
        input_val,
    )
    if dim is None:
        dim = []
    elif isinstance(dim, int):
        dim = [dim]

    min_out = impl.reduce.amin(ctx, target, source_ir, f"{name}_amax", abs_out, dim, keepdim)

    return cast_trt_tensor(ctx, min_out, trt.bool, f"{name}_cast_to_bool")


class ATenSliceTensorInputs(StrictlyTyped):
    """Input parameters for aten.slice.Tensor operation.

    Attributes:
        x (trt.ITensor): Input tensor
        dim (int): Dimension to slice along
        start (int | None, optional): Starting index. Defaults to None.
        end (int | None, optional): Ending index. Defaults to None.
        step (int, optional): Step size. Defaults to 1.
    """

    x: trt.ITensor
    dim: int
    start: int | None = None
    end: int | None = None
    step: int = 1

    @property
    def ndim(self) -> int:
        """Number of dimensions in input tensor."""
        return len(self.x.shape)

    @property
    def dim_size(self) -> int:
        """Size of slicing dimension."""
        return self.x.shape[self.dim]

    @property
    def start_as_dims(self) -> trt.Dims:
        """Starting index as TensorRT Dims."""
        if self.start is None:
            return trt.Dims([0])
        return trt.Dims([make_axis_nonnegative(self.start, dim_size=self.dim_size)])

    @property
    def shape_as_dims(self) -> trt.Dims:
        """Slice shape as TensorRT Dims."""
        start = self.start_as_dims[0]
        if self.end is None:
            return trt.Dims([self.dim_size - start])
        end = make_axis_nonnegative(self.end, dim_size=self.dim_size)
        return trt.Dims([end - start])

    @property
    def step_as_dims(self) -> trt.Dims:
        """Step size as TensorRT Dims."""
        return trt.Dims([self.step])

    @model_validator(mode="after")
    def ensure_slicing_dim_is_static(self) -> Self:
        """Validate that slicing dimension is static.

        Returns:
            Self: Self if validation passes

        Raises:
            AssertionError: If slicing dimension is dynamic
        """
        assert (
            -self.ndim <= self.dim < self.ndim and self.dim_size != -1
        ), "Slicing along dynamic dimension must be handled by TorchTRT converters"
        return self


@dynamo_tensorrt_converter(
    torch.ops.aten.slice.Tensor,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
@enforce_tensor_types(
    {
        0: (trt.ITensor,),
    }
)
def aten_ops_slice(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert PyTorch's aten.slice.Tensor operation to TensorRT.

    Args:
        ctx (ConversionContext): Conversion context
        target (Target): Target operation
        args (tuple[Argument, ...]): Positional arguments
        kwargs (dict[str, Argument]): Keyword arguments
        name (str): Layer name

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Sliced tensor
    """
    try:
        args_kwargs = dict(zip(ATenSliceTensorInputs.model_fields.keys(), args))
        args_kwargs.update(kwargs)
        inputs = ATenSliceTensorInputs(**args_kwargs)
    except (ValidationError, AssertionError) as e:
        logger.warning(f"{name} will fall back to default TorchTRT converter - {e}")
        return impl.slice.slice_op(
            ctx,
            target,
            SourceIR.ATEN,
            name,
            args[0],
            args_bounds_check(args, 1, replacement=0),
            args_bounds_check(args, 2, replacement=None),
            args_bounds_check(args, 3, replacement=None),
            args_bounds_check(args, 4, replacement=1),
        )
    layer = ctx.net.add_slice(inputs.x, inputs.start_as_dims, inputs.shape_as_dims, inputs.step_as_dims)
    layer.set_input(5, get_trt_tensor(ctx, np.array([inputs.dim]), f"{name}_axes"))
    set_layer_name(layer, target, name, SourceIR.ATEN)
    return layer.get_output(0)


@dynamo_tensorrt_converter(
    torch.ops.aten._to_copy.default,
    supports_dynamic_shapes=True,
    priority=ConverterPriority.HIGH,
)
def aten_ops_to_copy(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert PyTorch's aten._to_copy operation to TensorRT.

    Args:
        ctx (ConversionContext): Conversion context
        target (Target): Target operation
        args (tuple[Argument, ...]): Positional arguments
        kwargs (dict[str, Argument]): Keyword arguments
        name (str): Layer name

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: Copied tensor with specified dtype
    """
    return impl.cast.to_copy(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        kwargs.get("dtype", args[0].dtype),
        force_layer=True,
    )
