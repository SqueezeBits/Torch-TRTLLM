from torch.fx.node import Node

from ..nodes import ToCopy
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAmongInputs


class FuseConsecutiveToCopys(NodewiseOptimizationPass):
    """Fuse two consecutive _to_copy nodes."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (parent := ToCopy.specialize_from(node))
            and (children := [child for child_node in node.users if (child := ToCopy.specialize_from(child_node))])
        ):
            return {}
        return {child.node: ReplaceAmongInputs(occurrences_of=node, by=parent.this) for child in children}
