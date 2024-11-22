# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from ...types import SymInt
from .call_function_node import CallFunctionNode


class EmbeddingNode(CallFunctionNode):
    weight: Node
    indices: Node
    padding_idx: SymInt = -1
    scale_grad_by_freq: bool = False
    sparse: bool = False

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.embedding.default,)
