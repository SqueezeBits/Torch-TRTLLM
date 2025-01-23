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

import torch
from torch.fx import Node

from ...types import Number
from ..nodes import Div, GetAttr, Mul
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FuseReciprocalMul(NodewiseOptimizationPass):
    """Rewrite `(1 / y) * x` or `x * (1 / y)` as `x / y`."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((mul := Mul.specialize_from(node)) and (inputs := find_div_inputs_if_fusible_with(mul))):
            return {}
        graph = node.graph
        x, div = inputs
        with graph.inserting_before(node):
            if not (fused_div := Div.create_from_overloadpacket(graph, args=(x, div.other))):
                return {}
            inject_stack_trace_from(*((div, x) if isinstance(x, Node) else (div,)), to=fused_div)
        return {node: ReplaceAllUses(by=fused_div.node)}


def find_div_inputs_if_fusible_with(mul: Mul) -> tuple[Node | Number, Div] | None:
    """Find division inputs that can be fused with a multiplication node.

    Checks if either input to the multiplication is a division that can be fused with the multiplication.

    Args:
        mul (Mul): The multiplication node to check for fusion

    Returns:
        tuple[Node | Number, Div] | None: A tuple of (other operand, division node) if fusion is possible,
            None otherwise
    """
    lhs, rhs = mul.this, mul.other
    if div := get_fusible_div_from(lhs):
        return rhs, div
    if div := get_fusible_div_from(rhs):
        return lhs, div
    return None


def get_fusible_div_from(x: Node | Number) -> Div | None:
    """Check if an operand is a division node that can be fused.

    Args:
        x (Node | Number): The operand to check

    Returns:
        Div | None: The division node if it can be fused, None otherwise
    """
    if isinstance(x, Node) and (div := Div.specialize_from(x)) and is_equivalent_to_reciprocal(div):
        return div
    return None


def is_equivalent_to_reciprocal(div: Div) -> bool:
    """Check if a division node is equivalent to a reciprocal (1/x).

    Args:
        div (Div): The division node to check

    Returns:
        bool: True if the division is equivalent to a reciprocal, False otherwise
    """
    lhs = div.this
    if isinstance(lhs, Number) and lhs == 1:
        return True
    if isinstance(lhs, Node) and (get_attr := GetAttr.specialize_from(lhs)):
        return torch.all(get_attr.parameter == 1).item()
    return False
