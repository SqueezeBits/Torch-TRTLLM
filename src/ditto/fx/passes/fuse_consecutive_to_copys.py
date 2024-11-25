from torch.fx.node import Node

from ditto.fx.passes.node_wise_pass import NodewisePassResult

from ..nodes import ToCopy
from .node_wise_pass import NodewiseOptimizationPass, ReplaceAmongInputs


class FuseConsecutiveToCopys(NodewiseOptimizationPass):
    """Fuse two consecutive _to_copy nodes."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (parent := ToCopy.specialize_from(node))
            and (children := [child for child_node in node.users if (child := ToCopy.specialize_from(child_node))])
        ):
            return {}
        results: dict[Node, NodewisePassResult] = {}
        for child in children:
            child_node = child.node
            if stack_trace := child_node.stack_trace:
                child_node.stack_trace = f"{stack_trace}, pass: fused with {node} by {__name__}"
            results[child_node] = ReplaceAmongInputs(occurences_of=node, by=parent.this)
        return results
