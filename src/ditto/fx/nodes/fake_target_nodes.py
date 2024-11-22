from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from torch.fx import Node

from .call_function_node import CallFunctionNode


class ScaledDotProductAttentionNode(CallFunctionNode):
    query: Node
    key: Node
    value: Node
    attn_mask: Node | None = None
    dropout_p: float = 0.0
    is_causal: bool = False
    scale: float | None = None
    enable_gqa: bool = False

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch._C._nn.scaled_dot_product_attention,)

    @property
    def is_eligible_for_gpt_attention_plugin(self) -> bool:
        for name, field in self.model_fields.items():
            if field.is_required():
                continue
            if (value := getattr(self, name)) != (default_value := field.get_default()):
                logger.warning(
                    f"Cannot support the non-default '{name}={value}' provided to `F.scaled_dot_product_attention` "
                    f"(default is {default_value})"
                )
                return False
        return True
