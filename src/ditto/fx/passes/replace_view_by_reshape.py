from torch.fx import Node

from ..nodes import Reshape, View
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class ReplaceViewByReshape(NodewiseOptimizationPass):
    """A replacement for the `view_to_reshape` pass in TorchTRT for its shape inference error."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (view := View.specialize_from(node)):
            return {}
        with (graph := node.graph).inserting_after(node):
            reshape = Reshape.create(graph, view.this, view.size)
            inject_stack_trace_from(view, to=reshape)
        return {view.node: ReplaceAllUses(by=reshape.node)}
