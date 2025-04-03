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

from collections.abc import Callable
from typing import Any

from torch.fx import Node

from .call_function import FinalCallFunction
from modelopt.torch.quantization.tensor_quant import quantize_op

class ModelOptQuantizer(FinalCallFunction):
    input: Node
    amax: Node
    num_bits: int
    exponent_bits: int
    unsigned: bool
    narrow_range: bool

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (quantize_op.default,)

