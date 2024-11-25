from torch.fx.node import Node

from ..nodes import Reshape, SingleDimensionReshape
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAmongInputs


class FuseConsecutiveReshapes(NodewiseOptimizationPass):
    """Fuse two consecutive reshapes."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (parent := Reshape.specialize_from(node) or SingleDimensionReshape.specialize_from(node))
            and (
                children := [
                    child_reshape for child_node in node.users if (child_reshape := Reshape.specialize_from(child_node))
                ]
            )
        ):
            return {}
        results: dict[Node, NodewisePassResult] = {}
        for child_reshape in children:
            child_node = child_reshape.node
            if child_node.stack_trace:
                child_node.stack_trace = f"{child_node.stack_trace}, pass: fused with {node} by {__name__}"
            results[child_node] = ReplaceAmongInputs(occurences_of=node, by=parent.this)
        return results
