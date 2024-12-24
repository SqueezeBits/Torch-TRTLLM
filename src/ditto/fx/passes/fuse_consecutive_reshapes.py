from torch.fx.node import Node

from ..nodes import Reshape, SingleDimensionReshape
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAmongInputs


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
        return {child.node: ReplaceAmongInputs(occurrences_of=node, by=parent.this) for child in children}
