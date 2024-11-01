import logging

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from transformers import PretrainedConfig

from ...fake_targets import (
    FAKE_ROPE_TARGETS,
    ROPEConfig,
    get_llama2_rope_pattern_graph,
    get_llama2_rope_replacment_graph,
)
from ..utils import get_tensor_metadata
from .graph_pass import GraphOptimizationPass

logger = logging.getLogger(__name__)


class WrapRoPESubgraphs(GraphOptimizationPass):
    """Match and replace RoPE subgraphs by fake rope target (required for trtllm)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        if not (
            replaced_patterns := replace_pattern_with_filters(
                graph_module,
                pattern=get_llama2_rope_pattern_graph(),
                replacement=get_llama2_rope_replacment_graph(),
                ignore_literals=True,
            )
        ):
            return PassResult(graph_module, False)

        if not isinstance(
            pretrained_config := graph_module.meta.get("pretrained_config"),
            PretrainedConfig,
        ):
            logger.warning("No pretrained config found in graph module meta data")

        for replaced_pattern in replaced_patterns:
            node_map = {k.name: v for k, v in replaced_pattern.nodes_map.items()}
            rope_node: Node | None = None
            for node in replaced_pattern.replacements:
                if node.target in FAKE_ROPE_TARGETS:
                    rope_node = node
                    break
            else:
                continue

            if (x := node_map.get("x")) and (x_meta := get_tensor_metadata(x)):
                embed_dim = x_meta.shape[-1]
            else:
                logger.warning("Failed to infer `rotary_embedding_dim`")
                embed_dim = None

            rope_config = ROPEConfig.from_pretrained_config(
                pretrained_config,
                positional_embedding_type=FAKE_ROPE_TARGETS[rope_node.target],
                embedding_dim=embed_dim,
            )
            if output := replaced_pattern.nodes_map.get(replaced_pattern.anchor):
                rope_node.meta = output.meta
            rope_node.meta["rope_config"] = rope_config

        return PassResult(graph_module, True)
