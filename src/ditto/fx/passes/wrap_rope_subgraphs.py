from loguru import logger
from torch.fx import Node
from transformers import PretrainedConfig

from ..subgraphs import RoPESubgraph
from ..targets import (
    FAKE_ROPE_TARGETS,
    ROPEConfig,
)
from ..utils import get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class WrapRoPESubgraphs(NodewiseOptimizationPass):
    """Match and replace RoPE subgraphs by wrapped RoPE node (required for ReplaceSDPAByFakeGPTAttentionPlugin)."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (rope := RoPESubgraph.configure_from(node)):
            return {}

        graph = node.graph
        rope_target = FAKE_ROPE_TARGETS[rope.position_embedding_type]
        with graph.inserting_before(node):
            wrapped_rope = graph.call_function(rope_target, (rope.x, rope.cos, rope.sin))

        if x_meta := get_tensor_metadata(rope.x):
            embed_dim = x_meta.shape[-1]
        else:
            logger.warning("Failed to infer `rotary_embedding_dim`")
            embed_dim = None

        pretrained_config = None
        if (graph_module := graph.owning_module) is None or not isinstance(
            pretrained_config := graph_module.meta.get("pretrained_config"),
            PretrainedConfig,
        ):
            logger.warning("No pretrained config found in graph module meta data. Default RoPE config will be used.")

        rope_config = ROPEConfig.from_pretrained_config(
            pretrained_config,
            positional_embedding_type=rope.position_embedding_type,
            embedding_dim=embed_dim,
        )
        wrapped_rope.meta = rope.add.node.meta
        wrapped_rope.meta["rope_config"] = rope_config

        return {node: ReplaceAllUses(by=wrapped_rope)}
