from torch.fx import Node

from ..nodes.aten import BinaryElementwise
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class FixBinaryElementwiseOpOverloads(NodewiseOptimizationPass):
    """Fix binary elementwise operator overloads that failed to specialize.

    This pass attempts to re-specialize binary elementwise operators that has been incorrectly specialized
    by explicitly creating them from their overload packet. This handles cases where the initial
    specialization may have failed due to type mismatches or other issues.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        with node.graph.inserting_before(node):
            if not (
                node.target in BinaryElementwise.possible_targets()
                and BinaryElementwise.specialize_from(node) is None
                and (
                    replacement := BinaryElementwise.create_from_overloadpacket(
                        node.graph,
                        args=node.args,
                        kwargs=node.kwargs,
                        overloadpacket=node.target.overloadpacket,
                    )
                )
            ):
                return {}
        propagate_metadata_from(node, to=replacement)
        return {node: ReplaceAllUses(by=replacement.node)}
