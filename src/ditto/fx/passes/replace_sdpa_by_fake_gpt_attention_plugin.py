import tensorrt as trt
import torch
from loguru import logger
from torch.fx import GraphModule, Node

from ...debug import save_for_debug
from ...types import DataType
from ..nodes import Cat, GetAttr, Permute, Reshape, ScaledDotProductAttention, ToCopy
from ..subgraphs import Linear, TrailingReformatSequence
from ..targets import FAKE_ROPE_TARGETS, GPTAttentionPlugin, GPTAttentionPluginInputs, ROPEConfig
from ..utils import get_ancestors_with_depth, get_tensor_metadata
from .infra import GraphOptimizationPass, PassResult


class ReplaceSDPAByFakeGPTAttentionPlugin(GraphOptimizationPass):
    """Replace F.scaled_dot_product_attention by FakeGPTAttentionPlugin (required for trtllm)."""

    dtype: torch.dtype

    def call(self, graph_module: GraphModule) -> PassResult:
        save_for_debug("before_attn_plugin", graph_module)
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
                and (q_rope := TrailingReformatSequence.configure_from(sdpa.query).top).target in FAKE_ROPE_TARGETS
                and (k_rope_seq := TrailingReformatSequence.configure_from(sdpa.key))
                and (k_rope := k_rope_seq.top).target in FAKE_ROPE_TARGETS
                and (q_proj := find_projection(sdpa.query))
                and (k_proj := find_projection(sdpa.key))
                and (v_proj := find_projection(sdpa.value))
                and (q_proj.output == TrailingReformatSequence.configure_from(q_rope.all_input_nodes[0]).top)
                and (k_proj.output == TrailingReformatSequence.configure_from(k_rope.all_input_nodes[0]).top)
                and (v_seq := TrailingReformatSequence.configure_from(sdpa.value))
                and (v_proj.output == v_seq.top)
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
            if (num_key_groups := k_rope_seq.total_expansion) is None or (
                num_value_groups := v_seq.total_expansion
            ) is None:
                logger.error("Failed to infer `num_key_value_groups`.")
                continue
            if num_key_groups != num_value_groups:
                logger.error(
                    f"`num_key_value_groups` inferred from key ({num_key_groups=}) mismatched with "
                    f"the value inferred from value ({num_value_groups=})"
                )
                continue
            if num_kv_heads % (num_key_value_groups := num_key_groups) != 0:
                logger.error(f"{num_kv_heads=} is not a multiple of {num_key_value_groups=}")
                continue

            logger.debug(f"{num_key_value_groups=} at {layer_idx=}")

            if global_rope_config is None:
                global_rope_config = query_rope_config
            if global_plugin_inputs is None:
                rotary_inv_freq, rotary_cos_sin = (
                    torch.nn.Parameter(torch.from_numpy(x)) for x in global_rope_config.compute_rope_constants()
                )
                last_placeholder = graph.find_nodes(op="placeholder")[-1]
                with graph.inserting_after(last_placeholder):
                    _ = GetAttr.create(graph, "rotary_inv_freq", rotary_inv_freq)
                    _ = GetAttr.create(graph, "rotary_cos_sin", rotary_cos_sin)
                global_plugin_inputs = GPTAttentionPluginInputs.find_from(graph, global_rope_config.is_rope)
            if global_rope_config != query_rope_config:
                logger.warning(
                    f"rope config for query mismatched in GPTAttentionPlugin pattern {layer_idx}:\n"
                    f"  * global: {global_rope_config}\n"
                    f"  * query: {query_rope_config}\n"
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
                num_kv_heads=num_kv_heads // num_key_value_groups,
                layer_idx_in_cache_pool=layer_idx,
                head_size=embed_dim,
                type_id=DataType(self.dtype).to(trt.DataType),
                **global_rope_config.model_dump(),
            )
            with graph.inserting_before(node):
                qkv_nodes = (q_proj.output, k_proj.output, v_proj.output)
                qkv_cat = Cat.create(graph, qkv_nodes, -1)
                out_dtype: torch.dtype | None = None
                first_plugin_input: Node
                if isinstance(qkv_cat_output := qkv_cat.output, torch.Tensor) and qkv_cat_output.dtype != self.dtype:
                    first_plugin_input = ToCopy.create(graph, qkv_cat, dtype=self.dtype).node
                    out_dtype = qkv_cat_output.dtype
                else:
                    first_plugin_input = qkv_cat.node
                plugin_node = graph.call_function(
                    fake_gpt_attention_plugin,
                    (first_plugin_input,),
                    global_plugin_inputs.model_dump(),
                )
                if out_dtype is not None:
                    plugin_node = ToCopy.create(graph, plugin_node, dtype=out_dtype).node
                # See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
                # pylint: disable-next=invalid-name
                N, *others, Hq, _, Ev = sdpa_out_shape  # noqa: N806
                out_reshape = Reshape.create(graph, plugin_node, [N, *others, -1, Hq, Ev])
                dims = [*range(4 + len(others))]
                dims[-2], dims[-3] = dims[-3], dims[-2]
                out_permute = Permute.create(graph, out_reshape, dims)
            node.replace_all_uses_with(out_permute.node)
            modified = True
        return PassResult(graph_module=graph_module, modified=modified)


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
