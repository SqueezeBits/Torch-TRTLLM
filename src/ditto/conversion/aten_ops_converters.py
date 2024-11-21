# pylint: disable=unused-argument
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

from ..types import StrictlyTyped
from ..utils import make_axis_nonnegative


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
    return reduce_all(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        args_bounds_check(args, 1, replacement=None),
        args_bounds_check(args, 2, replacement=False),
    )


def reduce_all(
    ctx: ConversionContext,
    target: Target,
    source_ir: SourceIR | None,
    name: str,
    input_val: trt.ITensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> trt.ITensor:
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
    x: trt.ITensor
    dim: int
    start: int | None = None
    end: int | None = None
    step: int = 1

    @property
    def ndim(self) -> int:
        return len(self.x.shape)

    @property
    def dim_size(self) -> int:
        return self.x.shape[self.dim]

    @property
    def start_as_dims(self) -> trt.Dims:
        if self.start is None:
            return trt.Dims([0])
        return trt.Dims([make_axis_nonnegative(self.start, dim_size=self.dim_size)])

    @property
    def shape_as_dims(self) -> trt.Dims:
        start = self.start_as_dims[0]
        if self.end is None:
            return trt.Dims([self.dim_size - start])
        end = make_axis_nonnegative(self.end, dim_size=self.dim_size)
        return trt.Dims([end - start])

    @property
    def step_as_dims(self) -> trt.Dims:
        return trt.Dims([self.step])

    @model_validator(mode="after")
    def ensure_slicing_dim_is_static(self) -> Self:
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
    try:
        inputs = ATenSliceTensorInputs(
            **dict(zip(ATenSliceTensorInputs.model_fields.keys(), args))  # type: ignore[arg-type]
        )
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
