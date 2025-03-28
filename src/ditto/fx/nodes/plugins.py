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

from ..targets import (
    Fp8RowwiseGemmPlugin,
    GemmPlugin,
    GPTAttentionPlugin,
    WeightOnlyGroupwiseQuantMatmulPlugin,
    WeightOnlyQuantMatmulPlugin,
)
from .call_function import CallFunction


class Fp8RowwiseGemm(CallFunction):
    """A plugin specialization representing a gemm plugin node.

    Attributes:
        this (Node): The first input node
        other (Node): The second input node (expected to be a weight tensor)
        token_scaling (Node): The token scaling node
        channel_scaling (Node): The channel scaling node
    """

    this: Node
    other: Node
    token_scaling: Node
    channel_scaling: Node

    @property
    def target(self) -> Fp8RowwiseGemmPlugin:
        assert isinstance(t := super().target, Fp8RowwiseGemmPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, Fp8RowwiseGemmPlugin)


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
    """A plugin specialization representing a GPT attention plugin node.

    Attributes:
        qkv (Node): The input node for the query, key, and value projections
    """

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


class WeightOnlyQuantMatmul(CallFunction):
    """A plugin specialization representing a weight-only quantized matrix multiplication node.

    Attributes:
        this (Node): The first input node
        other (Node): The second input node (expected to be a weight tensor)
        scale (Node): The scale node
    """

    this: Node
    other: Node
    scale: Node

    @property
    def target(self) -> WeightOnlyQuantMatmulPlugin:
        assert isinstance(t := super().target, WeightOnlyQuantMatmulPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, WeightOnlyQuantMatmulPlugin)


class WeightOnlyGroupwiseQuantMatmul(CallFunction):
    """A plugin specialization representing a weight-only groupwise quantized matrix multiplication node.

    Attributes:
        this (Node): The first input node
        other (Node): The second input node (expected to be a weight tensor)
        scale (Node): The scale node
        zeros (Node | None): The zeros node
        bias (Node | None): The bias node
        alpha (Node | None): The alpha node
    """

    this: Node
    other: Node
    scale: Node
    zeros: Node | None
    bias: Node | None
    alpha: Node | None

    @property
    def target(self) -> WeightOnlyGroupwiseQuantMatmulPlugin:
        assert isinstance(t := super().target, WeightOnlyGroupwiseQuantMatmulPlugin)
        return t

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return ()

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return isinstance(node.target, WeightOnlyGroupwiseQuantMatmulPlugin)
