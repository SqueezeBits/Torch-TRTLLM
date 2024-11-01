import torch
from torch.fx import Node

from ..utils import get_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateUniqueClone(NodeWiseOptimizationPass):
    """Replace tensor copying ops if the original tensor is not used elsewhere."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (is_equivalent_to_clone(node) and len((input_node := node.all_input_nodes[0]).users) == 1):
            return {}
        return {node: input_node}


def is_equivalent_to_clone(node: Node) -> bool:
    return node.target is torch.ops.aten.clone.default or (
        node.target is torch.ops.aten._to_copy.default
        and (input_metadata := get_tensor_metadata(node.all_input_nodes[0])) is not None
        and isinstance(target_dtype := node.kwargs.get("dtype", None), torch.dtype)
        and target_dtype == input_metadata.dtype
    )
