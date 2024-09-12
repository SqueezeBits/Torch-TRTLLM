from collections.abc import Sequence

import tensorrt as trt
import torch
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo import SourceIR
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
    impl,
)
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    args_bounds_check,
    aten_ops_softmax,
    aten_ops_sub,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
)

dynamo_tensorrt_converter(torch.ops.aten.sub.default, supports_dynamic_shapes=False)(aten_ops_sub)
dynamo_tensorrt_converter(torch.ops.aten._safe_softmax.default, supports_dynamic_shapes=False)(aten_ops_softmax)

# dynamo_tensorrt_converter(torch._C._nn.scaled_dot_product_attention, supports_dynamic_shapes=True)(
#     tensorrt_scaled_dot_product_attention
# )


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
