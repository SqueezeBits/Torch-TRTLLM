from torch.fx import Node

from ..nodes import Combine
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateNopCatOrStack(NodeWiseOptimizationPass):
    """Eliminate cat or stack called with just one input tensor."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not ((cat_or_stack := Combine.specialize_from(node)) and len(cat_or_stack.tensors) == 1):
            return {}
        return {node: cat_or_stack.tensors[0]}
