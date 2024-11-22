from torch.fx import Node

from ..nodes import CatNode, StackNode
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateNopCatOrStack(NodeWiseOptimizationPass):
    """Eliminate cat or stack called with just one input tensor."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        cat_or_stack: CatNode | StackNode | None
        if not (
            (cat_or_stack := CatNode.specialize_from(node) or StackNode.specialize_from(node))
            and len(cat_or_stack.tensors) == 1
        ):
            return {}
        return {node: cat_or_stack.tensors[0]}
