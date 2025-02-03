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

import operator
from collections.abc import Sequence

import tensorrt as trt
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import dynamo_tensorrt_converter

from ..types import verify


@dynamo_tensorrt_converter(
    operator.getitem,
    supports_dynamic_shapes=True,
)
def convert_getitem(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> trt.ITensor | Sequence[trt.ITensor]:
    """Convert a getitem operator to TensorRT.

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
        len(args) == 2
        and (tensors := verify(args[0], as_type=list[trt.ITensor] | tuple[trt.ITensor, ...])) is not None
        and (index := verify(args[1], as_type=int)) is not None
    )
    return tensors[index]
