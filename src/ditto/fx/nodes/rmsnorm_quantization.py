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

from torch.fx.node import Node

from ..targets import RmsnormQuantizationPlugin
from .call_function import CallFunction


class RmsnormQuantization(CallFunction):
    """A plugin specialization representing a rmsnorm quantization plugin node.

    Attributes:
        this (Node): The first input node
        weight (Node): The second input node (expected to be a weight tensor)
        bias (Node): The third input node (expected to be a bias tensor)
        scale (Node): The fourth input node (expected to be a scale tensor)
        clamp_val (Node | None): The fifth input node (expected to be a clamp value tensor)
    """

    this: Node
    weight: Node
    bias: Node
    scale: Node
    clamp_val: Node | None = None

    @property
    def target(self) -> RmsnormQuantizationPlugin:
        assert isinstance(t := super().target, RmsnormQuantizationPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, RmsnormQuantizationPlugin)
