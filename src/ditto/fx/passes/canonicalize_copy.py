from torch.fx import Node

from ..nodes import Clone, ToCopy
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class CanonicalizeCopy(NodewiseOptimizationPass):
    """Eliminate or simplify copy-like ops."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if clone := Clone.specialize_from(node):
            return {node: ReplaceAllUses(by=clone.this)}

        if copy := ToCopy.specialize_from(node):
            if copy.dtype_unchanged:
                return {node: ReplaceAllUses(by=copy.this)}

            if len(node.kwargs) > 1 and "dtype" in node.kwargs:
                node.kwargs = {"dtype": node.kwargs["dtype"]}
                return {node: ModifiedInsideThePass()}

        return {}
