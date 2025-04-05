# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator

import tensorrt as trt
import torch
from loguru import logger
from pydantic import Field
from torch.fx import Graph, GraphModule, Node
from typing_extensions import Self

from ...configs import TRTLLMMapping
from ...debug import save_for_debug
from ...types import DataType, StrictlyTyped, SymbolicShape, expect_identical
from ..nodes import (
    BMM,
    Cat,
    GetAttr,
    GetItem,
    NodeSpecialization,
    Permute,
    Reshape,
    Rope,
    ScaledDotProductAttention,
    SplitWithSizes,
    SqueezeDim,
    ToCopy,
)
from ..subgraphs import FusedLinear, Linear, TrailingReformatPath
from ..targets import FAKE_ROPE_TARGETS, GPTAttentionPlugin, GPTAttentionPluginInputs, ROPEConfig
from ..utils import find_nearest, get_nodes_with_depth, get_tensor_metadata
from .infra import GraphOptimizationPass, PassResult
from .parallelize_linear import (
    parallelize_column_linear,
    parallelize_reformat,
    parallelize_row_linear,
)


class ReplaceSDPAByGPTAttentionPlugin(GraphOptimizationPass):
    """Replace F.scaled_dot_product_attention by GPTAttentionPlugin (required for trtllm).

    Attributes:
        dtype (torch.dtype): The data type of the input tensor
    """

    dtype: torch.dtype
    mapping: TRTLLMMapping = Field(frozen=True)

    # pylint: disable-next=too-many-locals,too-many-statements
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
                and sdpa.default_scale is not None
                and (attn := MHA.extract_from(sdpa) or MLA.extract_from(sdpa, self.mapping.tp_size))
            ):
                continue

            if not (
                sdpa.is_eligible_for_gpt_attention_plugin(is_mla_enabled=isinstance(attn, MLA))
                and attn.rope_config is not None
            ):
                continue
            if isinstance(attn, MLA):
                q_lora_rank = attn.q_lora_rank
                kv_lora_rank = attn.kv_lora_rank
                qk_nope_head_dim = attn.qk_nope_head_dim
                qk_rope_head_dim = attn.qk_rope_head_dim
                v_head_dim = attn.v_head_dim
            else:
                q_lora_rank = 0
                kv_lora_rank = 0
                qk_nope_head_dim = 0
                qk_rope_head_dim = 0
                v_head_dim = 0
            attn.rope_config.compute_rope_constants(qk_rope_head_dim)

            layer_idx += 1
            logger.debug(f"Replacing SDPA at layer {layer_idx} with {attn = }")
            if global_rope_config is None:
                global_rope_config = attn.rope_config
                global_rope_config.save_debug_artifacts()
            elif global_rope_config != attn.rope_config:
                logger.warning(
                    f"Local RoPE config mismatched with the global one at layer {layer_idx}:\n"
                    f"  * global: {global_rope_config}\n"
                    f"  * query: {attn.rope_config}\n"
                    "Will use the global RoPE config anyway."
                )

            if global_plugin_inputs is None:
                last_placeholder = list(graph.find_nodes(op="placeholder"))[-1]
                with graph.inserting_after(last_placeholder):
                    self.create_rope_input_nodes(
                        graph,
                        rotary_inv_freq=global_rope_config.rotary_inv_freq,
                        rotary_cos_sin=global_rope_config.rotary_cos_sin,
                        long_rope_rotary_inv_freq=global_rope_config.long_rope_rotary_inv_freq,
                        long_rope_rotary_cos_sin=global_rope_config.long_rope_rotary_cos_sin,
                    )
                global_plugin_inputs = GPTAttentionPluginInputs.find_from(graph, global_rope_config.is_rope)
                logger.debug(f"Found GPTAttentionPluginInputs for layer {layer_idx}")

            if isinstance(attn, MLA):
                attn.apply_lazy_tensor_parallelism(self.mapping)

            gpt_attention_plugin = GPTAttentionPlugin(
                layer_idx=layer_idx,
                num_heads=attn.num_heads,
                num_kv_heads=attn.num_kv_heads_per_group,
                layer_idx_in_cache_pool=layer_idx,
                head_size=attn.embed_dim,
                tp_size=self.mapping.tp_size,
                tp_rank=self.mapping.tp_rank,
                type_id=DataType(self.dtype).to(trt.DataType),
                q_scaling=sdpa.default_scale / sdpa.scale,
                is_mla_enabled=isinstance(attn, MLA),
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                **attn.rope_config.model_dump(),
            )

            with graph.inserting_before(node):
                plugin_inputs = global_plugin_inputs.model_dump()
                if isinstance(attn, MHA):
                    qkv = attn.qkv
                else:
                    if attn.is_deepseek_v2_lite:
                        k_pe = SqueezeDim.create(graph, attn.k_pe, 0)
                        qkv = Cat.create(graph, [attn.hidden_states, attn.compressed_kv, k_pe], 1).node
                        fused_q_proj, q_b_proj, kv_b_proj = attn.create_mla_weights(graph)
                    else:
                        compressed_q = attn.q_b_proj.mm.this
                        k_pe = SqueezeDim.create(graph, attn.k_pe, 0)
                        qkv = Cat.create(graph, [compressed_q, attn.compressed_kv, k_pe], 1).node
                        fused_q_proj, q_b_proj, kv_b_proj = attn.create_mla_weights(graph)
                    mla_inputs = {
                        "fused_q_proj": fused_q_proj,
                        "q_b_proj": q_b_proj,
                        "kv_b_proj": kv_b_proj,
                    }
                    plugin_inputs.update(mla_inputs)

                if (qkv_meta := get_tensor_metadata(qkv)) and ((out_dtype := qkv_meta.dtype) != self.dtype):
                    qkv = ToCopy.create(graph, qkv, dtype=self.dtype).node

                plugin_node = graph.call_function(
                    gpt_attention_plugin,
                    (qkv,),
                    plugin_inputs,
                )
                if out_dtype is not None:
                    plugin_node = ToCopy.create(graph, plugin_node, dtype=out_dtype).node
                # See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
                # pylint: disable-next=invalid-name
                N, *others, Hq, _, Ev = attn.output_shape  # noqa: N806
                out_reshape = Reshape.create(graph, plugin_node, [N, *others, -1, Hq // self.mapping.tp_size, Ev])
                dims = [*range(4 + len(others))]
                dims[-2], dims[-3] = dims[-3], dims[-2]
                out_permute = Permute.create(graph, out_reshape, dims)
            node.replace_all_uses_with(out_permute.node)
            modified = True
        return PassResult(
            graph_module=graph_module,
            modified=modified,
            require_fake_tensor_prop=isinstance(attn, MLA) and self.mapping.tp_size > 1,
        )

    def create_rope_input_nodes(
        self,
        graph: Graph,
        *,
        rotary_inv_freq: torch.nn.Parameter | None,
        rotary_cos_sin: torch.nn.Parameter | None,
        long_rope_rotary_inv_freq: torch.nn.Parameter | None,
        long_rope_rotary_cos_sin: torch.nn.Parameter | None,
    ) -> None:
        """Create GetAttr nodes for RoPE parameters in the graph.

        Creates GetAttr nodes to access RoPE (Rotary Position Embedding) parameters
        that will be used by the GPT attention plugin. Only creates nodes for
        parameters that are not None.

        Args:
            graph (Graph): The computation graph to add GetAttr nodes to
            rotary_inv_freq (torch.nn.Parameter | None): RoPE inverse frequency parameter
            rotary_cos_sin (torch.nn.Parameter | None): RoPE cosine/sine parameter
            long_rope_rotary_inv_freq (torch.nn.Parameter | None): Long RoPE inverse frequency parameter
            long_rope_rotary_cos_sin (torch.nn.Parameter | None): Long RoPE cosine/sine parameter

        Returns:
            None
        """
        for name, param in [
            ("rotary_inv_freq", rotary_inv_freq),
            ("rotary_cos_sin", rotary_cos_sin),
            ("long_rope_rotary_inv_freq", long_rope_rotary_inv_freq),
            ("long_rope_rotary_cos_sin", long_rope_rotary_cos_sin),
        ]:
            if param is not None:
                _ = GetAttr.create(graph, name, param)


class MHA(StrictlyTyped):
    """Multi-Head Attention configuration for GPTAttentionPlugin.

    Attributes:
        qkv (Node): The fused QKV projection node
        rope_config (ROPEConfig): The RoPE configuration
        num_attn_groups (int): The number of attention groups
        num_heads (int): The number of attention heads
        embed_dim (int): The embedding dimension
        num_kv_heads (int): The number of KV heads
        output_shape (SymbolicShape): The output shape
    """

    qkv: Node
    rope_config: ROPEConfig
    num_attn_groups: int
    num_heads: int
    embed_dim: int
    num_kv_heads: int
    output_shape: SymbolicShape

    @property
    def num_kv_heads_per_group(self) -> int:
        """Number of KV heads per attention group.

        Returns:
            int: The number of KV heads per attention group
        """
        return self.num_kv_heads // self.num_attn_groups

    @classmethod  # pylint: disable-next=too-many-locals
    def extract_from(cls, sdpa: ScaledDotProductAttention) -> Self | None:
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
            sdpa (ScaledDotProductAttention): The ScaledDotProductAttention node to analyze

        Returns:
            Self | None:
                If all requirements are met:
                    A MHA object with the extracted configuration
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
            q_rope.target in FAKE_ROPE_TARGETS.values()
            and k_rope.target in FAKE_ROPE_TARGETS.values()
            and (q_proj := find_nearest(Linear, sdpa.query))
            and (k_proj := find_nearest(Linear, sdpa.key))
            and (v_proj := find_nearest(Linear, sdpa.value))
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

        return cls(
            qkv=q_proj.output_node,
            rope_config=rope_config,
            num_attn_groups=num_attn_groups,
            num_heads=query.shape[-3],
            embed_dim=embed_dim,
            num_kv_heads=num_kv_heads,
            output_shape=(*query.shape[:-1], value.shape[-1]),
        )


# pylint: disable=too-many-locals
class MLA(StrictlyTyped):
    """Multi-head Latent Attention configuration for GPTAttentionPlugin.

    This class represents a multi-head latent attention mechanism that is introduced in DeepseekV2.

    Attributes:
        num_heads (int): Number of attention heads
        embed_dim (int): Dimension of the embeddings
        num_kv_heads (int): Number of key/value heads (for GQA)
        output_shape (SymbolicShape): Shape of the output tensor
        hidden_states (Node): Input hidden states node
        compressed_kv (Node): Compressed key/value states node
        k_pe (Node): Key position embedding node
        q_proj (Linear): Query projection layer
        kv_b_proj (Linear): Key-value projection layer B
        q_lora_rank (int): Rank of query LoRA weights
        kv_lora_rank (int): Rank of key-value LoRA weights
        qk_nope_head_dim (int): Head dimension for query/key without position embeddings
        qk_rope_head_dim (int): Head dimension for query/key with position embeddings
        v_head_dim (int): Head dimension for values
    """

    num_heads: int
    embed_dim: int
    num_kv_heads: int
    output_shape: SymbolicShape
    hidden_states: Node
    compressed_kv: Node
    k_pe: Node
    q_a_proj: Linear | None
    q_b_proj: Linear
    kv_a_proj: Linear
    kv_b_proj: Linear
    o_proj: Linear
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    is_deepseek_v2_lite: bool

    @property
    def num_kv_heads_per_group(self) -> int:
        """Returns number of KV heads per attention group.

        Returns:
            int: Always returns 1 for MLA
        """
        return 1

    @property
    def rope_config(self) -> ROPEConfig | None:
        """Get the RoPE configuration from the query projection output node.

        Searches for a RoPE node in the computation graph starting from the query projection
        output node and returns its configuration if found.

        Returns:
            ROPEConfig | None: The RoPE configuration if found, None otherwise
        """
        if not (rope := find_nearest(Rope, self.q_b_proj.output_node, follow_parent=False, follow_first_only=False)):
            return None
        return rope.meta.get("rope_config")

    @classmethod
    def extract_from(cls, sdpa: ScaledDotProductAttention, tp_size: int = 1) -> Self | None:
        """Extracts MLA configuration from a scaled dot product attention node.

        Analyzes the computation graph around the SDPA node to identify the MLA pattern
        and extracts relevant parameters and nodes.

        Args:
            sdpa (ScaledDotProductAttention): The SDPA node to analyze
            tp_size (int): Tensor parallelism size for model parallelism

        Returns:
            Self | None: MLA configuration if pattern matches, None otherwise
        """
        if not (
            (query := get_tensor_metadata(sdpa.query))
            and (key := get_tensor_metadata(sdpa.key))
            and (value := get_tensor_metadata(sdpa.value))
            and (num_kv_heads := expect_identical(key.shape[-3], value.shape[-3])) is not None
            and (
                q_b_proj := find_nearest(
                    Linear,
                    sdpa.query,
                    follow_first_only=False,
                    continue_if=lambda n: n.op == "call_function" and n.target in FAKE_ROPE_TARGETS.values(),
                )
            )
            and (
                q_split := find_nearest(
                    SplitWithSizes, q_b_proj.output_node, follow_parent=False, follow_first_only=False
                )
            )
            and (q_split_outputs := [getitem for user in q_split.users if (getitem := GetItem.specialize_from(user))])
            and len(q_split_outputs) == 2
            and (
                kv_b_proj := find_nearest(
                    Linear,
                    sdpa.key,
                    follow_first_only=False,
                    continue_if=lambda n: n.op == "call_function" and n.target in FAKE_ROPE_TARGETS.values(),
                )
            )
            and (
                kv_b_split := find_nearest(
                    SplitWithSizes, kv_b_proj.output_node, follow_parent=False, follow_first_only=False
                )
            )
            and (
                kv_b_split_outputs := [
                    getitem for user in kv_b_split.users if (getitem := GetItem.specialize_from(user))
                ]
            )
            and len(kv_b_split_outputs) == 2
            and (kv_a_proj := find_nearest(Linear, kv_b_proj.mm.this, follow_first_only=False))
            and (
                kv_a_split := find_nearest(
                    SplitWithSizes, kv_a_proj.output_node, follow_parent=False, follow_first_only=False
                )
            )
            and (
                kv_a_split_outputs := [
                    getitem for user in kv_a_split.users if (getitem := GetItem.specialize_from(user))
                ]
            )
            and len(kv_a_split_outputs) == 2
            and (o_proj := find_nearest(Linear, sdpa.value, follow_parent=False, follow_first_only=False))
            and get_tensor_metadata(q_split_outputs[0].node) is not None
            and get_tensor_metadata(q_split_outputs[1].node) is not None
            and get_tensor_metadata(kv_b_split_outputs[1].node) is not None
            and get_tensor_metadata(kv_a_split.this) is not None
        ):
            return None

        is_deepseek_v2_lite = q_b_proj.mm.this == kv_a_proj.mm.this
        q_a_proj = None
        if not is_deepseek_v2_lite:
            q_a_proj = find_nearest(Linear, q_b_proj.mm.this, follow_first_only=False)
            if q_a_proj is None or q_a_proj.mm.this != kv_a_proj.mm.this:
                raise NotImplementedError("Found MLA graph that is not supported yet.")
        hidden_states = kv_a_proj.mm.this
        compressed_kv = kv_b_proj.mm.this
        k_pe = kv_a_split_outputs[1].node
        q_lora_rank = q_b_proj.weight_tensor.shape[0]
        kv_lora_rank = kv_b_proj.weight_tensor.shape[0]
        qk_nope_head_dim = q_split_outputs[0].meta["val"].shape[-1]
        qk_rope_head_dim = q_split_outputs[1].meta["val"].shape[-1]
        v_head_dim = kv_b_split_outputs[1].meta["val"].shape[-1]
        embed_dim = kv_a_split.this.meta["val"].shape[-1]

        return cls(
            num_heads=query.shape[-3] // tp_size,
            embed_dim=embed_dim,
            num_kv_heads=num_kv_heads // tp_size,
            output_shape=(*query.shape[:-1], value.shape[-1]),
            hidden_states=hidden_states,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            q_a_proj=q_a_proj,
            q_b_proj=q_b_proj,
            kv_a_proj=kv_a_proj,
            kv_b_proj=kv_b_proj,
            o_proj=o_proj,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            is_deepseek_v2_lite=is_deepseek_v2_lite,
        )

    def apply_lazy_tensor_parallelism(self, mapping: TRTLLMMapping) -> None:
        """Apply tensor parallelism transformations lazily to the MLA layers.

        Tensor parallelism on MLA linears doesn't follow the original pattern.
        Instead,
            - q_a_proj, kv_a_proj: excluded from TP
            - q_b_proj, kv_b_proj: column parallel
            - o_proj: row parallel
        However, applying it in the original step(in the ParallelizeLinear pass) yields a fake tensor propagation error,
        because shape mismatches occur in the MLA layers that have not been wrapped into the GPTAttention plugin.
        Those shape mismatches of intermediate activations are meant to be handled inside the GPTAttention plugin.
        Therefore, we apply tensor parallelism on the MLA layers lazily while wrapping them into the plugin.

        Args:
            mapping (TRTLLMMapping): The tensor parallelism mapping configuration


        Returns:
            None
        """
        if mapping.tp_size == 1:
            return

        parallelize_column_linear(self.q_b_proj, mapping, inplace=True)
        parallelize_column_linear(self.kv_b_proj, mapping, inplace=True)
        parallelize_row_linear(self.o_proj, mapping)
        intermediate_nodes = get_nodes_with_depth(
            self.o_proj.mm.this, break_if=lambda n: ScaledDotProductAttention.specialize_from(n) is not None
        )
        for node in intermediate_nodes:
            parallelize_reformat(node, mapping)
            node.meta.pop("val", None)

    def create_mla_weights(self, graph: Graph) -> tuple[Node, Node, Node]:
        """Creates and processes MLA weight tensors.

        Performs necessary transformations on the projection weights,
        to prepare them for the GPTAttentionPlugin.

        Args:
            graph (Graph): The computation graph to add new operations to

        Returns:
            tuple[Node, Node, Node]: Tuple containing:
                - Fused query weight node
                - Query projection weight node
                - Concatenated key-value projection weight node
        """
        q_b_proj_weight: Node | NodeSpecialization = self.q_b_proj.weight_node
        kv_b_proj_weight: Node | NodeSpecialization = self.kv_b_proj.weight_node

        q_b_proj_weight_permute = Permute.create(graph, q_b_proj_weight, (1, 0))
        q_b_proj_weight = Reshape.create(
            graph,
            q_b_proj_weight_permute,
            (self.num_heads, (self.qk_nope_head_dim + self.qk_rope_head_dim), self.q_lora_rank),
        )
        q_b_proj_weight = SplitWithSizes.create(
            graph, q_b_proj_weight, [self.qk_nope_head_dim, self.qk_rope_head_dim], 1
        )
        q_nope_weight = graph.call_function(
            operator.getitem,
            (q_b_proj_weight.node, 0),
        )
        q_pe_weight = graph.call_function(
            operator.getitem,
            (q_b_proj_weight.node, 1),
        )

        kv_b_proj_weight = Permute.create(graph, kv_b_proj_weight, (1, 0))
        kv_b_proj_weight = Reshape.create(
            graph, kv_b_proj_weight, (self.num_kv_heads, (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank)
        )
        kv_b_proj_weight = SplitWithSizes.create(graph, kv_b_proj_weight, [self.qk_nope_head_dim, self.v_head_dim], 1)
        k_nope_weight: Node | NodeSpecialization = graph.call_function(
            operator.getitem,
            (kv_b_proj_weight.node, 0),
        )
        v_weight = graph.call_function(
            operator.getitem,
            (kv_b_proj_weight.node, 1),
        )

        k_nope_weight_reshaped = Reshape.create(
            graph, k_nope_weight, (self.num_kv_heads * self.qk_nope_head_dim, self.kv_lora_rank)
        )
        v_weight_reshaped = Reshape.create(graph, v_weight, (self.num_kv_heads * self.v_head_dim, self.kv_lora_rank))
        kv_b_proj_weight = Cat.create(graph, [k_nope_weight_reshaped, v_weight_reshaped], 0)

        k_nope_weight = Permute.create(graph, k_nope_weight, (0, 2, 1))
        fused_q_weight: Node | NodeSpecialization = BMM.create(graph, k_nope_weight, q_nope_weight)
        fused_q_weight = Cat.create(graph, [fused_q_weight, q_pe_weight], 1)
        fused_q_weight = Reshape.create(
            graph, fused_q_weight, (self.num_kv_heads * (self.kv_lora_rank + self.qk_rope_head_dim), self.q_lora_rank)
        )

        return fused_q_weight.node, q_b_proj_weight_permute.node, kv_b_proj_weight.node
