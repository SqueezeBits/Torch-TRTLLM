import torch
from loguru import logger
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.shape_prop import TensorMetadata

from ...constants import GPT_ATTENTION_PLUGIN_DTYPE
from ..nodes import ScaledDotProductAttention
from ..subgraphs import Linear
from ..targets import FAKE_ROPE_TARGETS, GPTAttentionPlugin, GPTAttentionPluginInputs, ROPEConfig
from ..utils import get_ancestors_with_depth, get_tensor_metadata, populate_tensor_metadata, traceback_reformats
from .graph_pass import GraphOptimizationPass


class ReplaceSDPAByFakeGPTAttentionPlugin(GraphOptimizationPass):
    """Replace F.scaled_dot_product_attention by FakeGPTAttentionPlugin (required for trtllm)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        layer_idx = -1
        graph = graph_module.graph
        global_rope_config: ROPEConfig | None = None
        global_plugin_inputs: GPTAttentionPluginInputs | None = None

        modified = False
        for node in graph.nodes:
            if not (
                (sdpa := ScaledDotProductAttention.specialize_from(node))
                and sdpa.is_eligible_for_gpt_attention_plugin
                and (query := get_tensor_metadata(sdpa.query))
                and (key := get_tensor_metadata(sdpa.key))
                and (value := get_tensor_metadata(sdpa.value))
                and (q_rope := traceback_reformats(sdpa.query)).target in FAKE_ROPE_TARGETS
                and (k_rope := traceback_reformats(sdpa.key)).target in FAKE_ROPE_TARGETS
                and (q_proj := find_projection(sdpa.query))
                and (k_proj := find_projection(sdpa.key))
                and (v_proj := find_projection(sdpa.value))
                and (q_proj.mm == traceback_reformats(q_rope.all_input_nodes[0]))
                and (k_proj.mm == traceback_reformats(k_rope.all_input_nodes[0]))
                and (v_proj.mm == traceback_reformats(sdpa.value))
            ):
                continue

            layer_idx += 1
            num_heads = query.shape[-3]
            sdpa_out_shape = (*query.shape[:-1], value.shape[-1])
            if (num_kv_heads := key.shape[-3]) != value.shape[-3]:
                logger.error(
                    "The input key and value with different number of heads is not supported. "
                    f"(key.shape: {key.shape}, value.shape: {value.shape})"
                )
                continue
            if not (embed_dim := query.shape[-1]) == key.shape[-1] == value.shape[-1]:
                logger.error(
                    "The input query, key and value with different embedding dimensions is not supported. "
                    f"(query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape})"
                )
                continue
            if not isinstance(query_rope_config := q_rope.meta.get("rope_config"), ROPEConfig):
                logger.error(f"rope config for query not found in GPTAttentionPlugin pattern {layer_idx}")
                continue
            if not isinstance(key_rope_config := k_rope.meta.get("rope_config"), ROPEConfig):
                logger.error(f"rope config for key not found in GPTAttentionPlugin pattern {layer_idx}")
                continue

            if global_rope_config is None:
                global_rope_config = query_rope_config
            if global_plugin_inputs is None:
                rotary_inv_freq, rotary_cos_sin = (
                    torch.nn.Parameter(torch.from_numpy(x)) for x in global_rope_config.compute_rope_constants()
                )
                graph_module.register_parameter("rotary_inv_freq", rotary_inv_freq)
                graph_module.register_parameter("rotary_cos_sin", rotary_cos_sin)
                last_placeholder = graph.find_nodes(op="placeholder")[-1]
                with graph.inserting_after(last_placeholder):
                    populate_tensor_metadata(graph.get_attr("rotary_inv_freq"), rotary_inv_freq)
                    populate_tensor_metadata(graph.get_attr("rotary_cos_sin"), rotary_cos_sin)
                global_plugin_inputs = GPTAttentionPluginInputs.find_from(graph, global_rope_config.is_rope)
            if global_rope_config != query_rope_config:
                logger.warning(
                    f"rope config for key mismatched in GPTAttentionPlugin pattern {layer_idx}:\n"
                    f"  * global: {global_rope_config}\n"
                    f"  * key: {query_rope_config}\n"
                    "Will use the global rope config anyway."
                )
            if global_rope_config != key_rope_config:
                logger.warning(
                    f"rope config for key mismatched in GPTAttentionPlugin pattern {layer_idx}:\n"
                    f"  * global: {global_rope_config}\n"
                    f"  * key: {key_rope_config}\n"
                    "Will use the global rope config anyway."
                )

            fake_gpt_attention_plugin = GPTAttentionPlugin(
                layer_idx=layer_idx,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=embed_dim,
                **global_rope_config.model_dump(),
            )
            with graph.inserting_before(node):
                qkv_cat = graph.call_function(torch.ops.aten.cat.default, ((q_proj.mm, k_proj.mm, v_proj.mm), -1))
                (q, k, v) = (get_tensor_metadata(x) for x in (q_proj.mm, k_proj.mm, v_proj.mm))
                prev_metadata: TensorMetadata | None = None
                if q and k and v:
                    qkv_cat_shape = torch.Size((*q.shape[:-1], sum(x.shape[-1] for x in (q, k, v))))
                    prev_metadata = populate_tensor_metadata(qkv_cat, q, shape=qkv_cat_shape)
                out_dtype: torch.dtype | None = None
                if prev_metadata and len(prev_metadata.shape) == 3 and prev_metadata.shape[0] == 1:
                    qkv_cat = graph.call_function(torch.ops.aten.squeeze.dim, (qkv_cat, 0))
                    prev_metadata = populate_tensor_metadata(qkv_cat, prev_metadata, shape=prev_metadata.shape[1:])
                if prev_metadata and prev_metadata.dtype != GPT_ATTENTION_PLUGIN_DTYPE:
                    qkv_cat = graph.call_function(
                        torch.ops.aten._to_copy.default, (qkv_cat,), {"dtype": GPT_ATTENTION_PLUGIN_DTYPE}
                    )
                    out_dtype = prev_metadata.dtype
                    prev_metadata = populate_tensor_metadata(qkv_cat, prev_metadata, dtype=GPT_ATTENTION_PLUGIN_DTYPE)
                plugin_node = graph.call_function(
                    fake_gpt_attention_plugin, (qkv_cat, *global_plugin_inputs.model_dump().values())
                )
                if q:
                    prev_metadata = populate_tensor_metadata(plugin_node, q, dtype=GPT_ATTENTION_PLUGIN_DTYPE)
                if out_dtype is not None:
                    plugin_node = graph.call_function(
                        torch.ops.aten._to_copy.default,
                        (plugin_node,),
                        {"dtype": out_dtype},
                    )
                    if prev_metadata:
                        prev_metadata = populate_tensor_metadata(plugin_node, prev_metadata, dtype=out_dtype)

                # See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
                # pylint: disable-next=invalid-name
                N, *others, Hq, L, Ev = sdpa_out_shape  # noqa: N806
                out_reshape = graph.call_function(
                    torch.ops.aten.reshape.default, (plugin_node, [N, *others, -1, Hq, Ev])
                )
                if prev_metadata:
                    prev_metadata = populate_tensor_metadata(out_reshape, prev_metadata, shape=(N, *others, L, Hq, Ev))
                ndim = 4 + len(others)
                dims = [*range(ndim)]
                dims[-2], dims[-3] = dims[-3], dims[-2]
                output = graph.call_function(torch.ops.aten.permute.default, (out_reshape, dims))
                if prev_metadata:
                    prev_metadata = populate_tensor_metadata(output, prev_metadata, shape=sdpa_out_shape)

            node.replace_all_uses_with(output)
            modified = True
        return PassResult(graph_module, modified)


def find_projection(x: Node) -> Linear | None:
    if not (
        ancester_linear_subgraphs := {
            subgraph: depth
            for node, depth in get_ancestors_with_depth(x).items()
            if (subgraph := Linear.configure_from(node))
        }
    ):
        return None
    return min(ancester_linear_subgraphs, key=lambda subgraph: ancester_linear_subgraphs[subgraph])
