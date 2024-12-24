from torch.fx import Node

from ..nodes import Cat, Stack, Unsqueeze
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateNopCatOrStack(NodewiseOptimizationPass):
    """Eliminate cat or stack called with just one input tensor."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (cat := Cat.specialize_from(node)) and len(cat.tensors) == 1:
            return {node: ReplaceAllUses(by=cat.tensors[0])}

        if (stack := Stack.specialize_from(node)) and len(stack.tensors) == 1:
            x = stack.tensors[0]
            graph = x.graph
            with graph.inserting_after(x):
                unsqueeze = Unsqueeze.create(graph, x, stack.dim)
            return {node: ReplaceAllUses(by=unsqueeze.node, propagate_meta=True)}

        return {}
