from loguru import logger
from torch.fx import GraphModule

from transformers import PretrainedConfig

from ..subgraphs import RoPESubgraph
from ..targets import (
    FAKE_ROPE_TARGETS,
    ROPEConfig,
)
from ..utils import get_tensor_metadata
from .infra import GraphOptimizationPass, PassResult


class WrapRoPESubgraphs(GraphOptimizationPass):
    """Match and replace RoPE subgraphs by wrapped RoPE node (required for ReplaceSDPAByFakeGPTAttentionPlugin)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        if not isinstance(
            pretrained_config := graph_module.meta.get("pretrained_config"),
            PretrainedConfig,
        ):
            logger.warning("No pretrained config found in graph module meta data")

        for node in graph.nodes:
            if not (rope := RoPESubgraph.configure_from(node)):
                continue

            graph = node.graph
            rope_target = FAKE_ROPE_TARGETS[rope.position_embedding_type]
            with graph.inserting_before(node):
                wrapped_rope = graph.call_function(rope_target, (rope.x, rope.cos, rope.sin))

            if x_meta := get_tensor_metadata(rope.x):
                embed_dim = x_meta.shape[-1]
            else:
                logger.warning("Failed to infer `rotary_embedding_dim`")
                embed_dim = None
            rope_config = ROPEConfig.from_pretrained_config(
                pretrained_config,
                positional_embedding_type=rope.position_embedding_type,
                embedding_dim=embed_dim,
            )
            wrapped_rope.meta = rope.add.node.meta
            wrapped_rope.meta["rope_config"] = rope_config

            node.replace_all_uses_with(wrapped_rope)
            modified = True

        return PassResult(graph_module=graph_module, modified=modified)
