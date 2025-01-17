from typing import Any, TypeVar, overload

import tensorrt as trt
import torch
from loguru import logger
from torch.fx import GraphModule
from typing_extensions import Self

from ...debug import save_for_debug
from ...types import DataType, StrictlyTyped, SymbolicShape
from ..nodes import GetAttr, Permute, Reshape, ScaledDotProductAttention, ToCopy
from ..subgraphs import FusedLinear, TrailingReformatPath
from ..subgraphs.linear import Linear, find_nearest_linear_projection
from ..targets import FAKE_ROPE_TARGETS, GPTAttentionPlugin, GPTAttentionPluginInputs, ROPEConfig
from ..utils import get_tensor_metadata
from .infra import GraphOptimizationPass, PassResult


class ReplaceSDPAByFakeGPTAttentionPlugin(GraphOptimizationPass):
    """Replace F.scaled_dot_product_attention by FakeGPTAttentionPlugin (required for trtllm)."""

    dtype: torch.dtype

    # pylint: disable-next=too-many-locals
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
                and (qkv_proj_and_mha := MHAConfig.extract_from(sdpa))
            ):
                continue

            qkv_proj, mha = qkv_proj_and_mha
            layer_idx += 1
            logger.debug(f"Replacing SDPA at layer {layer_idx} with {mha = }")

            if global_rope_config is None:
                global_rope_config = mha.rope_config
            elif global_rope_config != mha.rope_config:
                logger.warning(
                    f"Local RoPE config mismatched with the global one at layer {layer_idx}:\n"
                    f"  * global: {global_rope_config}\n"
                    f"  * query: {mha.rope_config}\n"
                    "Will use the global RoPE config anyway."
                )

            if global_plugin_inputs is None:
                logger.debug(f"Computing RoPE constants for layer {layer_idx}")
                rotary_inv_freq, rotary_cos_sin = (
                    torch.nn.Parameter(torch.from_numpy(x)) for x in global_rope_config.compute_rope_constants()
                )
                last_placeholder = list(graph.find_nodes(op="placeholder"))[-1]
                with graph.inserting_after(last_placeholder):
                    _ = GetAttr.create(graph, "rotary_inv_freq", rotary_inv_freq)
                    _ = GetAttr.create(graph, "rotary_cos_sin", rotary_cos_sin)
                global_plugin_inputs = GPTAttentionPluginInputs.find_from(graph, global_rope_config.is_rope)
                logger.debug(f"Found GPTAttentionPluginInputs for layer {layer_idx}")

            fake_gpt_attention_plugin = GPTAttentionPlugin(
                layer_idx=layer_idx,
                num_heads=mha.num_heads,
                num_kv_heads=mha.num_kv_heads_per_group,
                layer_idx_in_cache_pool=layer_idx,
                head_size=mha.embed_dim,
                type_id=DataType(self.dtype).to(trt.DataType),
                **mha.rope_config.model_dump(),
            )
            with graph.inserting_before(node):
                qkv = qkv_proj.output_node
                if (qkv_meta := get_tensor_metadata(qkv)) and ((out_dtype := qkv_meta.dtype) != self.dtype):
                    qkv = ToCopy.create(graph, qkv, dtype=self.dtype).node
                plugin_node = graph.call_function(
                    fake_gpt_attention_plugin,
                    (qkv,),
                    global_plugin_inputs.model_dump(),
                )
                if out_dtype is not None:
                    plugin_node = ToCopy.create(graph, plugin_node, dtype=out_dtype).node
                # See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
                # pylint: disable-next=invalid-name
                N, *others, Hq, _, Ev = mha.output_shape  # noqa: N806
                out_reshape = Reshape.create(graph, plugin_node, [N, *others, -1, Hq, Ev])
                dims = [*range(4 + len(others))]
                dims[-2], dims[-3] = dims[-3], dims[-2]
                out_permute = Permute.create(graph, out_reshape, dims)
            node.replace_all_uses_with(out_permute.node)
            modified = True
        return PassResult(graph_module=graph_module, modified=modified)


class MHAConfig(StrictlyTyped):
    """Multi-Head Attention configuration."""

    rope_config: ROPEConfig
    num_attn_groups: int
    num_heads: int
    embed_dim: int
    num_kv_heads: int
    output_shape: SymbolicShape

    @property
    def num_kv_heads_per_group(self) -> int:
        """Number of KV heads per attention group."""
        return self.num_kv_heads // self.num_attn_groups

    @classmethod  # pylint: disable-next=too-many-locals
    def extract_from(cls, sdpa: ScaledDotProductAttention) -> tuple[Linear, Self] | None:
        """Analyze a scaled dot product attention node to extract QKV projection and MHA configuration.

        This method examines a scaled dot product attention (SDPA) node to identify and validate:

        1. The fused linear projection that generates Q, K, V tensors
        2. The RoPE (Rotary Position Embedding) configuration
        3. The MHA configuration including:
           - Number of attention heads
           - Number of KV heads
           - Number of attention groups (for grouped query attention)
           - Embedding dimension
           - Output shape

        Requirements for successful extraction:
        - Q, K, V projections must come from the same fused linear layer
        - Q and K must have RoPE embeddings applied
        - Q and K must share identical RoPE configurations
        - K and V must have matching expansion ratios for GQA
        - Number of KV heads must be divisible by number of attention groups

        Args:
            sdpa: The ScaledDotProductAttention node to analyze

        Returns:
            If all requirements are met:
                A tuple containing:
                - The fused linear projection node that generates Q,K,V
                - A MHAConfig object with the extracted configuration
            If any requirement fails:
                None
        """
        if not (
            (query := get_tensor_metadata(sdpa.query))
            and (key := get_tensor_metadata(sdpa.key))
            and (value := get_tensor_metadata(sdpa.value))
            and (embed_dim := expect_identical(query.shape[-1], key.shape[-1], value.shape[-1])) is not None
            and (num_kv_heads := expect_identical(key.shape[-3], value.shape[-3])) is not None
        ):
            return None

        q_seq = TrailingReformatPath.configure_from(sdpa.query)
        k_seq = TrailingReformatPath.configure_from(sdpa.key)
        v_seq = TrailingReformatPath.configure_from(sdpa.value)
        q_rope = q_seq.top
        k_rope = k_seq.top
        q = TrailingReformatPath.configure_from(q_rope.all_input_nodes[0]).top
        k = TrailingReformatPath.configure_from(k_rope.all_input_nodes[0]).top
        v = v_seq.top
        if not (
            q_rope.target in FAKE_ROPE_TARGETS
            and k_rope.target in FAKE_ROPE_TARGETS
            and (q_proj := find_nearest_linear_projection(sdpa.query))
            and (k_proj := find_nearest_linear_projection(sdpa.key))
            and (v_proj := find_nearest_linear_projection(sdpa.value))
            and q_proj.output_node == k_proj.output_node == v_proj.output_node
            and (fused_linear := FusedLinear.configure_from(q_proj.mm.node))
            and tuple(s.node for s in fused_linear.slices) == (q, k, v)
            and (
                rope_config := expect_identical(
                    q_rope.meta.get("rope_config"),
                    k_rope.meta.get("rope_config"),
                    expecting_type=ROPEConfig,
                )
            )
            and (num_attn_groups := expect_identical(k_seq.total_expansion, v_seq.total_expansion)) is not None
            and (num_kv_heads % num_attn_groups == 0)
        ):
            return None

        return q_proj, cls(
            rope_config=rope_config,
            num_attn_groups=num_attn_groups,
            num_heads=query.shape[-3],
            embed_dim=embed_dim,
            num_kv_heads=num_kv_heads,
            output_shape=(*query.shape[:-1], value.shape[-1]),
        )


T = TypeVar("T")


@overload
def expect_identical(value: T, *others: T) -> T | None:
    ...


@overload
def expect_identical(value: Any, *others: Any, expecting_type: type[T]) -> T | None:
    ...


def expect_identical(value: Any, *others: Any, expecting_type: type[T] | None = None) -> T | None:
    """Compare multiple values for equality and optionally check their type.

    Args:
        value: First value to compare
        *others: Additional values to compare against the first value
        expecting_type: Optional type to validate all values against. If None, no type checking is performed.

    Returns:
        The first value if all values are equal and match the specified type (if provided).
        None if any values are not equal or don't match the type.
    """
    if expecting_type is not None and not all(isinstance(v, expecting_type) for v in (value, *others)):
        return None
    if all(v == value for v in others):
        return value
    return None
