from torch.fx import Node

from ...types import Number
from ..nodes import Binary
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult


class HerdConstantsToTheRight(NodewiseOptimizationPass):
    """Herd constant inputs of binary nodes to the right hand side."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (binary := Binary.specialize_from(node))
            and binary.is_commutative
            and isinstance(binary.this, Number)
            and len(node.args) >= 2
        ):
            return {}
        node.args = (node.args[1], node.args[0], *node.args[2:])
        return {node: ModifiedInsideThePass()}
