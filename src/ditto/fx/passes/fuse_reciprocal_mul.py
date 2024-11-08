import torch
from torch.fx import Node

from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import DivNode, GetAttrNode, MulNode, Number


class FuseReciprocalMul(NodeWiseOptimizationPass):
    """Rewrite `(1 / y) * x` or `x * (1 / y)` as `x / y`."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not ((mul := MulNode.specialize_from(node)) and (inputs := find_div_inputs_if_fusible_with(mul))):
            return {}
        graph = node.graph
        x, div = inputs
        with graph.inserting_before(node):
            fused_div = graph.call_function(torch.ops.aten.div.Tensor, (x, div.y))
            if t := get_tensor_metadata(node):
                populate_tensor_metadata(fused_div, t)
            if stack_trace := node.stack_trace:
                fused_div.stack_trace = (
                    f"{stack_trace}, pass: {node} and {div.node} fused by {FuseReciprocalMul.__name__}"
                )
        return {node: fused_div}


def find_div_inputs_if_fusible_with(mul: MulNode) -> tuple[Node | Number, DivNode] | None:
    lhs, rhs = mul.x, mul.y
    if div := get_fusible_div_from(lhs):
        return rhs, div
    if div := get_fusible_div_from(rhs):
        return lhs, div
    return None


def get_fusible_div_from(x: Node | Number) -> DivNode | None:
    if isinstance(x, Node) and (div := DivNode.specialize_from(x)) and is_equivalent_to_reciprocal(div):
        return div
    return None


def is_equivalent_to_reciprocal(div: DivNode) -> bool:
    lhs = div.x
    if isinstance(lhs, Number) and lhs == 1:
        return True
    if isinstance(lhs, Node) and (getattr := GetAttrNode.specialize_from(lhs)):
        return torch.all(getattr.parameter, 1).item()  # type: ignore[return-value]
    return False
