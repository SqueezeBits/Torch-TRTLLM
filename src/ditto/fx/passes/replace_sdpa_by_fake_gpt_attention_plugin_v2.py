import operator
from functools import reduce

import torch
from loguru import logger
from tensorrt_llm.functional import PositionEmbeddingType
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from ...config import GPT_ATTENTION_PLUGIN_DTYPE
from ..nodes import ScaledDotProductAttention
from ..targets import GPTAttentionPlugin, GPTAttentionPluginInputs, ROPEConfig
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .graph_pass import GraphOptimizationPass


class ReplaceSDPAByFakeGPTAttentionPluginV2(GraphOptimizationPass):
    """Replace F.scaled_dot_product_attention by FakeGPTAttentionPlugin (required for trtllm)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        layer_idx = -1
        graph = graph_module.graph
        global_plugin_inputs: GPTAttentionPluginInputs | None = None

        modified = False
        for node in graph.nodes:
            if not (
                (sdpa := ScaledDotProductAttention.specialize_from(node))
                and sdpa.is_eligible_for_gpt_attention_plugin
                and (query := get_tensor_metadata(sdpa.query))
                and (key := get_tensor_metadata(sdpa.key))
                and (value := get_tensor_metadata(sdpa.value))
            ):
                continue

            layer_idx += 1
            ndim = len(query.shape)
            permutation = [*range(ndim - 3), ndim - 2, ndim - 3, ndim - 1]
            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            # pylint: disable-next=invalid-name
            N, *others, Hq, L, E = query.shape  # noqa: N806
            # pylint: disable-next=invalid-name
            H, _, Ev = value.shape[-3:]  # noqa: N806
            sdpa_out_shape = (N, *others, Hq, L, Ev)
            if not (E == Ev and H == Hq):
                logger.error(
                    "The query, key and value with different embedding dimensions or number of heads "
                    "are not supported yet. "
                    f"(query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape})"
                )
                continue

            if global_plugin_inputs is None:
                rope_config = ROPEConfig(
                    position_embedding_type=PositionEmbeddingType.learned_absolute,
                    rotary_embedding_dim=0,
                    rotary_embedding_base=0.0,
                    rotary_embedding_scale=0.0,
                    rotary_embedding_max_positions=4096,
                )
                global_plugin_inputs = GPTAttentionPluginInputs.find_from(graph, is_rope=rope_config.is_rope)

            fake_gpt_attention_plugin = GPTAttentionPlugin(
                layer_idx=layer_idx,
                num_heads=Hq,
                num_kv_heads=H,
                head_size=E,
                **rope_config.model_dump(),
            )

            def reformat_to_2d(x: Node, dims: list[int]) -> Node:
                transpose = graph.call_function(torch.ops.aten.permute.default, (x, dims))
                assert (t := get_tensor_metadata(x)) is not None
                t = populate_tensor_metadata(transpose, t, shape=(*t.shape[:-3], t.shape[-2], t.shape[-3], t.shape[-1]))
                symbolic_shape = (
                    reduce(operator.mul, t.shape[:-2], 1),
                    reduce(operator.mul, t.shape[-2:], 1),
                )
                shape = [-1 if isinstance(x, torch.SymInt) else x for x in symbolic_shape]
                y = graph.call_function(torch.ops.aten.reshape.default, (transpose, shape))
                populate_tensor_metadata(y, t, shape=symbolic_shape)
                return y

            with graph.inserting_before(node):
                query_2d, key_2d, value_2d = (
                    reformat_to_2d(x, permutation) for x in (sdpa.query, sdpa.key, sdpa.value)
                )
                assert (q := get_tensor_metadata(query_2d)) is not None
                assert (k := get_tensor_metadata(key_2d)) is not None
                assert (v := get_tensor_metadata(value_2d)) is not None
                qkv_cat = graph.call_function(torch.ops.aten.cat.default, ((query_2d, key_2d, value_2d), -1))
                qkv_cat_shape = torch.Size((*q.shape[:-1], sum(x.shape[-1] for x in (q, k, v))))
                prev_metadata = populate_tensor_metadata(qkv_cat, query, shape=qkv_cat_shape)
                out_dtype: torch.dtype | None = None
                if len(prev_metadata.shape) == 3 and prev_metadata.shape[0] == 1:
                    qkv_cat = graph.call_function(torch.ops.aten.squeeze.dim, (qkv_cat, 0))
                    prev_metadata = populate_tensor_metadata(qkv_cat, prev_metadata, shape=prev_metadata.shape[1:])
                if prev_metadata.dtype != GPT_ATTENTION_PLUGIN_DTYPE:
                    qkv_cat = graph.call_function(
                        torch.ops.aten._to_copy.default, (qkv_cat,), {"dtype": GPT_ATTENTION_PLUGIN_DTYPE}
                    )
                    out_dtype = prev_metadata.dtype
                    prev_metadata = populate_tensor_metadata(qkv_cat, prev_metadata, dtype=GPT_ATTENTION_PLUGIN_DTYPE)
                plugin_node = graph.call_function(
                    fake_gpt_attention_plugin,
                    (qkv_cat, *(x for x in global_plugin_inputs.model_dump().values() if x is not None)),
                )
                prev_metadata = populate_tensor_metadata(plugin_node, q, dtype=GPT_ATTENTION_PLUGIN_DTYPE)
                if out_dtype is not None:
                    plugin_node = graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (plugin_node,),
                        {"dtype": out_dtype},
                    )
                    prev_metadata = populate_tensor_metadata(plugin_node, prev_metadata, dtype=out_dtype)

                # See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
                # pylint: disable-next=invalid-name
                # N, *others, Hq, L, Ev = sdpa_out_shape  # noqa: N806
                out_reshape = graph.call_function(
                    torch.ops.aten.reshape.default, (plugin_node, [N, *others, -1, Hq, Ev])
                )
                prev_metadata = populate_tensor_metadata(out_reshape, prev_metadata, shape=(N, *others, L, Hq, Ev))
                output = graph.call_function(torch.ops.aten.permute.default, (out_reshape, permutation))
                _ = populate_tensor_metadata(output, prev_metadata, shape=sdpa_out_shape)

            node.replace_all_uses_with(output)
            modified = True
        return PassResult(graph_module, modified)
