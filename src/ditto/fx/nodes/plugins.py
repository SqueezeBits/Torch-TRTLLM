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

from ..targets import GemmPlugin, GPTAttentionPlugin
from .call_function import CallFunction


class Gemm(CallFunction):
    """A plugin specialization representing a gemm plugin node.

    Attributes:
        this (Node): The first input node
        other (Node): The second input node (expected to be a weight tensor)
    """

    this: Node
    other: Node

    @property
    def target(self) -> GemmPlugin:
        assert isinstance(t := super().target, GemmPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, GemmPlugin)


class GPTAttention(CallFunction):
    """A plugin specialization representing a GPT attention plugin node."""

    model_config = {"extra": "ignore"}

    qkv: Node

    @property
    def target(self) -> GPTAttentionPlugin:
        assert isinstance(t := super().target, GPTAttentionPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, GPTAttentionPlugin)
