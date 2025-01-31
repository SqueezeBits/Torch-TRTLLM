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

# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch

from .aten_op import FinalATenOp
from .unary import Unary


class UnaryElementwise(Unary):
    """Base class for unary elementwise operations."""


@UnaryElementwise.register(torch.ops.aten.neg.default)
class Neg(UnaryElementwise, FinalATenOp):
    """Elementwise negation operation."""


@UnaryElementwise.register(torch.ops.aten.sqrt.default)
class Sqrt(UnaryElementwise, FinalATenOp):
    """Elementwise square root operation."""
