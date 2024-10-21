import logging
from typing import TypeVar

import torch
from tensorrt_llm.functional import PositionEmbeddingType
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications
from transformers import PretrainedConfig

from ...fake_gpt_attention_plugin import FakeGPTAttentionPlugin, ROPEConfig

logger = logging.getLogger(__name__)


def replace_sdpa_by_fake_gpt_attention_plugin(graph_module: GraphModule) -> GraphModule:
    # TODO: add more pattern matching logics for other ROPE with types other than `rope_gpt_neox`.
    replaced_patterns = replace_pattern_with_filters(
        graph_module,
        pattern=get_gpt_attention_plugin_pattern(),
        replacement=get_gpt_attention_plugin_replacement(),
        ignore_literals=True,
    )
    if (
        isinstance(
            pretrained_config := graph_module.meta.get("pretrained_config"),
            PretrainedConfig,
        )
        and replaced_patterns
    ):
        for replaced_pattern in replaced_patterns:
            node_map = {k.name: v for k, v in replaced_pattern.nodes_map.items()}
            plugin_node: Node | None = None
            for node in replaced_pattern.replacements:
                if node.target is FakeGPTAttentionPlugin:
                    plugin_node = node
                    break
            else:
                continue

            rope_config = ROPEConfig()
            rope_config.position_embedding_type = PositionEmbeddingType.rope_gpt_neox
            rope_config.rotary_embedding_base = lookup_attributes(
                pretrained_config,
                "rope_theta",
                default=rope_config.rotary_embedding_base,
            )
            if (query := node_map.get("query")) is not None and isinstance(
                query_meta := query.meta.get("tensor_meta"), TensorMetadata
            ):
                rope_config.rotary_embedding_dim = query_meta.shape[-1]
            else:
                logger.warning("Failed to infer `rotary_embedding_dim`")
            rope_config.rotary_embedding_max_positions = lookup_attributes(
                pretrained_config,
                "max_position_embeddings",
                default=rope_config.rotary_embedding_max_positions,
            )
            plugin_node.meta["rope_config"] = rope_config
    clean_up_graph_after_modifications(graph_module)
    return graph_module


def get_gpt_attention_plugin_pattern() -> Graph:
    graph = Graph()
    query = graph.placeholder("query")
    key = graph.placeholder("key")
    value = graph.placeholder("value")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")

    query = insert_rotary_pos_emb_subgraph(graph, query, cos, sin)
    key = insert_rotary_pos_emb_subgraph(graph, key, cos, sin)

    output = graph.call_function(
        torch._C._nn.scaled_dot_product_attention,
        (query, key, value, None, 0.0, False),
    )
    _ = graph.output((output,))
    graph.lint()
    return graph


def get_gpt_attention_plugin_replacement() -> Graph:
    graph = Graph()
    query = graph.placeholder("query")
    key = graph.placeholder("key")
    value = graph.placeholder("value")
    _ = graph.placeholder("cos")
    _ = graph.placeholder("sin")

    qkv = graph.call_function(torch.ops.aten.cat.default, ((query, key, value), -3))
    batch_size = graph.call_function(torch.ops.aten.sym_size.int, (query, 0))
    query_shape = graph.call_function(torch.ops.aten.sym_size.default, (query,))
    qkv_2d = graph.call_function(torch.ops.aten.reshape.default, (qkv, (batch_size, -1)))
    fake_plugin = graph.call_function(
        FakeGPTAttentionPlugin,
        (qkv_2d,),
    )
    output = graph.call_function(torch.ops.aten.reshape.default, (fake_plugin, query_shape))
    _ = graph.output((output,))
    graph.lint()
    return graph


def insert_rotary_pos_emb_subgraph(
    graph: Graph,
    x: Node,
    cos: Node,
    sin: Node,
    *,
    axis: int = -1,
    embed_dim: int = 128,
) -> Node:
    x_cos = graph.call_function(torch.ops.aten.mul.Tensor, (x, cos))
    # Note: integer literals used in the slice nodes will be considered as wild cards by the subgraph matcher
    x_slice_0 = graph.call_function(torch.ops.aten.slice.Tensor, (x, axis, 0, embed_dim // 2))
    x_slice_1 = graph.call_function(torch.ops.aten.slice.Tensor, (x, axis, embed_dim // 2, (1 << 63) - 1))
    neg_x_slice_1 = graph.call_function(torch.ops.aten.neg.default, (x_slice_1,))
    rotated_x = graph.call_function(torch.ops.aten.cat.default, ((neg_x_slice_1, x_slice_0), axis))
    rotated_x_sin = graph.call_function(torch.ops.aten.mul.Tensor, (rotated_x, sin))
    return graph.call_function(torch.ops.aten.add.Tensor, (x_cos, rotated_x_sin))


T = TypeVar("T")


def lookup_attributes(pretrained_config: PretrainedConfig, *names: str, default: T) -> T:
    for name in names:
        if hasattr(pretrained_config, name):
            return getattr(pretrained_config, name)
    return default
