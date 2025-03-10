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

import torch
from torch.fx import Node

from ..targets import Dequantize as DequantizeTarget
from .call_function import FinalCallFunction
from .get_attr import GetAttr


class Dequantize(FinalCallFunction):
    """A representation of the dequantization operation.

    Attributes:
        qweight (Node): The quantized weight tensor node.
        scale (Node): The scale tensor node.
        zeros (Node | None): The zeros tensor node, if any. Defaults to None.
    """

    qweight: Node
    scale: Node
    zeros: Node | None

    @property
    def target(self) -> DequantizeTarget:
        assert isinstance(t := super().target, DequantizeTarget)
        return t

    @property
    def qweight_tensor(self) -> torch.Tensor | None:
        if attr := GetAttr.specialize_from(self.qweight):
            return attr.tensor
        return None

    @property
    def scale_tensor(self) -> torch.Tensor | None:
        if attr := GetAttr.specialize_from(self.scale):
            return attr.tensor
        return None

    @property
    def zeros_tensor(self) -> torch.Tensor | None:
        if self.zeros is not None and (attr := GetAttr.specialize_from(self.zeros)):
            return attr.tensor
        return None

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, DequantizeTarget)
