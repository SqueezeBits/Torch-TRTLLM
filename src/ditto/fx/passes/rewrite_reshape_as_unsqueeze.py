import torch
from torch.fx import Node

from ..nodes import Reshape
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass


class RewriteReshapeAsUnsqueeze(NodeWiseOptimizationPass):
    """Rewrite reshape as unsqueeze if possible."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (reshape := Reshape.specialize_from(node))
            and (input_tensor := get_tensor_metadata(reshape.this))
            and (target_shape := reshape.target_shape) is not None
            and (dim := find_unsqueeze_dim(input_tensor.shape, target_shape)) is not None
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            unsqueeze = graph.call_function(torch.ops.aten.unsqueeze.default, (reshape.this, dim))
            if node.stack_trace:
                unsqueeze.stack_trace = f"{node.stack_trace}, pass: rewritten by {__name__}"
            if t := get_tensor_metadata(node):
                populate_tensor_metadata(unsqueeze, t)
        return {node: unsqueeze}


def find_unsqueeze_dim(
    input_shape: torch.Size,
    target_shape: torch.Size,
) -> int | None:
    ndim = len(input_shape)
    if ndim + 1 != len(target_shape):
        return None
    for i in range(ndim + 1):
        if torch.Size((*input_shape[:i], 1, *input_shape[i:])) == target_shape:
            return i
    return None
