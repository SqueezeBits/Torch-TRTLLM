import torch
from torch.fx import Node

from ..nodes import Index
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class RewriteIndexAsSingleSlice(NodewiseOptimizationPass):
    """Rewrite index op as single slice op when possible (required to support models with MQA)."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (index := Index.specialize_from(node)) and index.can_replace_with_single_slice:
            graph = node.graph

            dim = index.dim
            start = index.idx
            end = start + 1

            with graph.inserting_before(node):
                equivalent_slice = graph.call_function(torch.ops.aten.slice.Tensor, (index.this, dim, start, end))
                inject_stack_trace_from(node, to=equivalent_slice)

            return {node: ReplaceAllUses(by=equivalent_slice)}
        return {}
