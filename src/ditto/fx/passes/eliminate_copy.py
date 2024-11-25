import torch
from torch.fx import Node

from ..utils import get_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateCopy(NodeWiseOptimizationPass):
    """Replace tensor copying ops if the original tensor is not used elsewhere."""

    def rewrite(self, node: Node) -> dict[Node, Node]:
        if not is_equivalent_to_copy(node):
            return {}
        return {node: node.all_input_nodes[0]}


def is_equivalent_to_copy(node: Node) -> bool:
    return node.target is torch.ops.aten.clone.default or (
        node.target is torch.ops.aten._to_copy.default
        and (input_metadata := get_tensor_metadata(node.all_input_nodes[0])) is not None
        and isinstance(target_dtype := node.kwargs.get("dtype", None), torch.dtype)
        and target_dtype == input_metadata.dtype
    )
