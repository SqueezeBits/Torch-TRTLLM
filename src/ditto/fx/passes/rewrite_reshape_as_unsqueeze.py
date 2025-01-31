import torch
from torch.fx import Node

from ..nodes import Reshape, Unsqueeze
from ..utils import get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class RewriteReshapeAsUnsqueeze(NodewiseOptimizationPass):
    """Rewrite reshape as unsqueeze if possible."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (reshape := Reshape.specialize_from(node))
            and (input_tensor := get_tensor_metadata(reshape.this))
            and (output_tensor := get_tensor_metadata(reshape.node)) is not None
            and (dim := find_unsqueeze_dim(input_tensor.shape, output_tensor.shape)) is not None
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            unsqueeze = Unsqueeze.create(graph, reshape.this, dim)
            propagate_metadata_from(reshape, to=unsqueeze)
        return {node: ReplaceAllUses(by=unsqueeze.node)}


def find_unsqueeze_dim(
    input_shape: torch.Size,
    target_shape: torch.Size,
) -> int | None:
    """Find the dimension to unsqueeze to match the target shape.

    Args:
        input_shape (torch.Size): The input shape
        target_shape (torch.Size): The target shape

    Returns:
        int | None: The dimension to unsqueeze, or None if no match is found
    """
    ndim = len(input_shape)
    if ndim + 1 != len(target_shape):
        return None
    for i in range(ndim + 1):
        if torch.Size((*input_shape[:i], 1, *input_shape[i:])) == target_shape:
            return i
    return None
