from torch.fx.node import Node

from ..subgraphs.lora import MultiLora
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class StashLoraSubgraphs(NodewiseOptimizationPass):
    """Match and replace Lora subgraphs by a single Lora plugin node."""

    @property
    def reversed_traversal(self) -> bool:
        return True

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((multi_lora := MultiLora.configure_from(node)) and multi_lora.all_loras_unseen):
            return {}
        multi_lora.set_free_lora_proto()
        return {multi_lora.output_node: ReplaceAllUses(by=multi_lora.pre_lora_output_node)}
