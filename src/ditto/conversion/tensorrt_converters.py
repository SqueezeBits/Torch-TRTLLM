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

# mypy: disallow-untyped-decorators=False
# pylint: disable=unused-argument, duplicate-code

from collections.abc import Sequence

import tensorrt as trt
import torch
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import dynamo_tensorrt_converter
from torch_tensorrt.fx.converters.converter_utils import set_layer_name

from ..fx.targets import Quantize
from ..types import DataType, verify


@dynamo_tensorrt_converter(
    Quantize,
    supports_dynamic_shapes=True,
)
def convert_quantize(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a quantize operator to TensorRT.

    Args:
        ctx (ConversionContext): The conversion context
        target (Target): The operator target to convert
        args (tuple[Argument, ...]): Positional arguments to the operator
        kwargs (dict[str, Argument]): Keyword arguments to the operator
        name (str): Name for the TensorRT layer

    Returns:
        trt.ITensor | Sequence[trt.ITensor]: The indexed tensor from the input sequence
    """
    assert (
        isinstance(target, Quantize)
        and len(args) == 3
        and (input_tensor := verify(args[0], as_type=trt.ITensor)) is not None
        and (scale_tensor := verify(args[1], as_type=trt.ITensor)) is not None
        and (output_dtype := verify(args[2], as_type=torch.dtype)) is not None
    )

    quantize_layer = ctx.net.add_quantize(input_tensor, scale_tensor, DataType(output_dtype).to(trt.DataType))
    # quantize_layer.set_output_type(0, DataType(output_dtype).to(trt.DataType))
    set_layer_name(quantize_layer, target, name + "_quantize", SourceIR.ATEN)

    return quantize_layer.get_output(0)
