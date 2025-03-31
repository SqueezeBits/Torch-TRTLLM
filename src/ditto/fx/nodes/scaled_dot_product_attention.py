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

import math
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from torch.fx import Node

from ..utils import get_tensor_metadata
from .call_function import FinalCallFunction


class ScaledDotProductAttention(FinalCallFunction):
    """A representation of the scaled dot-product attention operation.

    This class encapsulates the scaled dot-product attention function, including its
    parameters and metadata, and provides utilities to analyze its compatibility with
    specific plugins or implementations.

    Attributes:
        query (Node): The query tensor node in the attention computation.
        key (Node): The key tensor node in the attention computation.
        value (Node): The value tensor node in the attention computation.
        attn_mask (Node | None): The attention mask node, if any. Defaults to None.
        dropout_p (float): The dropout probability applied during attention computation.
            Defaults to 0.0.
        is_causal (bool): Whether the attention is causal (e.g., masking future tokens).
            Defaults to False.
        scale (float | None): An optional scaling factor for the dot-product attention.
            Defaults to None.
        enable_gqa (bool): Whether to enable grouped-query attention (GQA). Defaults to False.
    """

    query: Node
    key: Node
    value: Node
    attn_mask: Node | None = None
    dropout_p: float = 0.0
    is_causal: bool = False
    scale: float
    enable_gqa: bool = False

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        """Get the possible target functions for scaled dot-product attention."""
        return (torch._C._nn.scaled_dot_product_attention,)

    @property
    def default_scale(self) -> float | None:
        """Get the default scaling factor for the attention scores.

        Returns:
            float | None: The default scaling factor if head size can be determined,
                         None otherwise.
        """
        head_size = q.shape[-1] if (q := get_tensor_metadata(self.query)) else None
        return None if head_size is None else 1 / math.sqrt(head_size)

    def is_eligible_for_gpt_attention_plugin(self, is_mla_enabled: bool = False) -> bool:
        """Check if the operation is eligible for the GPT attention plugin.

        Evaluates the compatibility of the scaled dot-product attention operation with
        the GPT attention plugin. This includes checks for default parameter values
        and specific scale settings.
        """
        for name, field in self.model_fields.items():
            if field.is_required():
                continue
            if (value := getattr(self, name)) != (default_value := field.get_default()):
                if is_mla_enabled:
                    continue
                if name == "scale" and self.default_scale and math.isclose(value, self.default_scale):
                    continue
                logger.warning(
                    f"Cannot support the non-default '{name}={value}' provided to `F.scaled_dot_product_attention` "
                    f"(default is {default_value})"
                )
                return False
        return True
