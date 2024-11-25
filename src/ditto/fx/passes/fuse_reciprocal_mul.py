import torch
from torch.fx import Node

from ...types import Number
from ..nodes import Div, GetAttr, Mul
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseReciprocalMul(NodewiseOptimizationPass):
    """Rewrite `(1 / y) * x` or `x * (1 / y)` as `x / y`."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((mul := Mul.specialize_from(node)) and (inputs := find_div_inputs_if_fusible_with(mul))):
            return {}
        graph = node.graph
        x, div = inputs
        with graph.inserting_before(node):
            fused_div = graph.call_function(torch.ops.aten.div.Tensor, (x, div.other))
            if t := get_tensor_metadata(node):
                populate_tensor_metadata(fused_div, t)
            if stack_trace := node.stack_trace:
                fused_div.stack_trace = (
                    f"{stack_trace}, pass: {node} and {div.node} fused by {FuseReciprocalMul.__name__}"
                )
        return {node: ReplaceAllUses(by=fused_div)}


def find_div_inputs_if_fusible_with(mul: Mul) -> tuple[Node | Number, Div] | None:
    lhs, rhs = mul.this, mul.other
    if div := get_fusible_div_from(lhs):
        return rhs, div
    if div := get_fusible_div_from(rhs):
        return lhs, div
    return None


def get_fusible_div_from(x: Node | Number) -> Div | None:
    if isinstance(x, Node) and (div := Div.specialize_from(x)) and is_equivalent_to_reciprocal(div):
        return div
    return None


def is_equivalent_to_reciprocal(div: Div) -> bool:
    lhs = div.this
    if isinstance(lhs, Number) and lhs == 1:
        return True
    if isinstance(lhs, Node) and (get_attr := GetAttr.specialize_from(lhs)):
        return torch.all(get_attr.parameter, 1).item()  # type: ignore[return-value]
    return False
