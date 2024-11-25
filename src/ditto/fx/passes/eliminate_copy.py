from torch.fx import Node

from ..nodes import CopyLike
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateCopy(NodewiseOptimizationPass):
    """Replace tensor copying ops if the original tensor is not used elsewhere."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((copy := CopyLike.specialize_from(node)) and copy.is_pure_copy):
            return {}
        return {node: ReplaceAllUses(by=copy.this)}
