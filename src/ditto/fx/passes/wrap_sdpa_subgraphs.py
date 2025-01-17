from torch.fx.node import Node

from ..subgraphs import ScaledDotProductAttentionSubgraph
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class WrapSDPASubgraphs(NodewiseOptimizationPass):
    """Match and replace scaled dot product attention subgraphs by a single `F.scaled_dot_product_attention` node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (sdpa := ScaledDotProductAttentionSubgraph.configure_from(node)) and (output := sdpa.insert_fused_graph())
        ):
            return {}
        return {sdpa.av_bmm.node: ReplaceAllUses(by=output)}
