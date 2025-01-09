from enum import IntEnum, IntFlag
from typing import Any, TypeVar

import numpy as np
import tensorrt as trt
import torch
from loguru import logger
from pydantic import Field
from tensorrt_llm.functional import (
    AttentionMaskType,
    PositionEmbeddingType,
    QuantMode,
    RopeEmbeddingUtils,
    RotaryScalingType,
)
from tensorrt_llm.plugin.plugin import ContextFMHAType
from torch.fx import Graph, Node
from transformers import PretrainedConfig
from typing_extensions import Self

from ...debug import open_debug_artifact
from ...types import StrictlyTyped
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin_field_types import PLUGIN_FIELD_TYPES


class Llama3ScalingConfig(StrictlyTyped):
    """Configuration for Llama 3's RoPE scaling parameters.

    Contains scaling factors for position embeddings used in Llama 3's attention mechanism.
    """

    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


# pylint: disable-next=too-many-instance-attributes
class ROPEConfig(StrictlyTyped):
    """Configuration for RoPE (Rotary Position Embedding) parameters.

    Contains parameters for RoPE used in the attention mechanism.
    """

    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.learned_absolute
    rotary_embedding_dim: int = 0
    rotary_embedding_base: float = 10000.0
    rotary_embedding_scale_type: RotaryScalingType = RotaryScalingType.none
    rotary_embedding_scale: float = 1.0
    rotary_embedding_short_m_scale: float = 1.0
    rotary_embedding_long_m_scale: float = 1.0
    rotary_embedding_max_positions: int = 1024
    rotary_embedding_original_max_positions: int = 1024
    llama3_scaling_config: Llama3ScalingConfig = Field(default_factory=Llama3ScalingConfig, exclude=True)

    @property
    def is_rope(self) -> bool:
        """Check if the RoPE configuration is valid."""
        return self.position_embedding_type.is_rope() and self.rotary_embedding_dim != 0

    @classmethod
    def from_pretrained_config(
        cls,
        pretrained_config: PretrainedConfig | None = None,
        positional_embedding_type: PositionEmbeddingType | None = None,
        embedding_dim: int | None = None,
    ) -> Self:
        """Create a RoPE configuration from a HuggingFace pretrained model configuration.

        This method extracts RoPE (Rotary Position Embedding) parameters from a HuggingFace
        pretrained model configuration and creates a corresponding ROPEConfig object.

        Args:
            pretrained_config: HuggingFace pretrained model configuration object. If None,
                default values will be used.
            positional_embedding_type: Override the position embedding type. If None, uses
                the type from pretrained_config.
            embedding_dim: Override the embedding dimension. If None, uses the dimension
                from pretrained_config.

        Returns:
            A ROPEConfig object initialized with parameters from the pretrained config,
            with any provided overrides applied.

        The method extracts the following parameters if available:
            - rope_theta: Base value for rotary embeddings
            - max_position_embeddings: Maximum sequence length
            - rope_scaling: Dictionary containing scaling parameters including:
                - factor: Scaling factor for rotary embeddings
                - type/rope_type: Type of rotary scaling to apply
                - Additional parameters for Llama3 scaling
        """
        rope_config = cls()
        rope_config.rotary_embedding_base = lookup_attributes(
            pretrained_config,
            "rope_theta",
            default=rope_config.rotary_embedding_base,
        )
        rope_config.rotary_embedding_max_positions = lookup_attributes(
            pretrained_config,
            "max_position_embeddings",
            default=rope_config.rotary_embedding_max_positions,
        )
        rope_scaling: dict[str, Any] = lookup_attributes(
            pretrained_config,
            "rope_scaling",
            default={},
            not_found_ok=True,
        )
        if rope_scaling:
            rope_config.rotary_embedding_scale = rope_scaling.get("factor", rope_config.rotary_embedding_scale)
            rope_config.rotary_embedding_original_max_positions = 1024  # TODO: need to be updated for long_rope type
            rotary_scaling_type: str | None = rope_scaling.get("type", rope_scaling.get("rope_type", None))
            if rotary_scaling_type is not None:
                rope_config.rotary_embedding_scale_type = RotaryScalingType.from_string(rotary_scaling_type)
            rope_config.llama3_scaling_config = Llama3ScalingConfig(**rope_scaling)
        if positional_embedding_type is not None:
            rope_config.position_embedding_type = positional_embedding_type
        if embedding_dim is not None:
            rope_config.rotary_embedding_dim = embedding_dim
        return rope_config

    def compute_rope_constants(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute RoPE constants for attention plugin.

        Returns:
            A tuple containing:
            - rotary_inv_freq: Inverse frequency tensor for RoPE
            - embed_positions: Pre-computed position embeddings
        """
        # TODO: replace this by `Attention.create_attention_const_params`
        rotary_inv_freq, embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            self.rotary_embedding_max_positions,
            self.rotary_embedding_dim,
            self.rotary_embedding_base,
            self.rotary_embedding_scale,
            self.rotary_embedding_scale_type,
            self.llama3_scaling_config.model_dump(),
        )
        with open_debug_artifact("rope_inputs.pt", "wb") as f:
            if f:
                torch.save(
                    {
                        "rotary_inv_freq": torch.from_numpy(rotary_inv_freq),
                        "rotary_cos_sin": torch.from_numpy(embed_positions),
                    },
                    f,
                )
        with open_debug_artifact("rope_config.json") as f:
            if f:
                f.write(self.model_dump_json(indent=2))
        return rotary_inv_freq, embed_positions


T = TypeVar("T")


def lookup_attributes(
    pretrained_config: PretrainedConfig | None,
    *names: str,
    default: T,
    not_found_ok: bool = False,
) -> T:
    """Look up attributes from a pretrained config, falling back to default if not found.

    Args:
        pretrained_config: Config object to look up attributes from
        *names: Attribute names to search for
        default: Default value to return if attributes not found
        not_found_ok: If True, suppress warning when attributes not found

    Returns:
        Found attribute value or default if not found
    """
    if pretrained_config is None:
        return default
    for name in names:
        if hasattr(pretrained_config, name):
            return getattr(pretrained_config, name)
    if not not_found_ok:
        logger.warning(
            "None of the following attributes are found in pretrained config. "
            f"Will use the default value {default}: {', '.join(names)}"
        )
    return default


class GPTAttentionPluginFields(StrictlyTyped):
    """Configuration fields for GPT attention plugin.

    Contains all the parameters needed to configure the GPT attention plugin,
    including attention dimensions, RoPE settings, and various optimization flags.
    """

    # the order of the attributes does matter!
    layer_idx: int
    num_heads: int
    vision_start: int = -1
    vision_length: int = -1
    num_kv_heads: int
    layer_idx_in_cache_pool: int
    head_size: int  # this field is actually `hidden_size_per_head`
    unidirectional: int = 1
    q_scaling: float = 1.0
    attn_logit_softcapping_scale: float = 0.0
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.learned_absolute
    rotary_embedding_dim: int = 0
    rotary_embedding_base: float = 10000.0
    rotary_embedding_scale_type: RotaryScalingType = RotaryScalingType.none
    rotary_embedding_scale: float = 1.0
    rotary_embedding_short_m_scale: float = 1.0
    rotary_embedding_long_m_scale: float = 1.0
    rotary_embedding_max_positions: int = 1024
    rotary_embedding_original_max_positions: int = 1024
    tp_size: int = 1
    tp_rank: int = 0
    unfuse_qkv_gemm: bool = False
    context_fmha_type: ContextFMHAType = ContextFMHAType.enabled
    kv_cache_quant_mode: QuantMode = QuantMode(0)
    remove_input_padding: bool = True
    mask_type: AttentionMaskType = AttentionMaskType.causal
    block_sparse_block_size: int = 64
    block_sparse_homo_head_pattern: bool = False
    block_sparse_num_local_blocks: int = 16
    block_sparse_vertical_stride: int = 8
    paged_kv_cache: bool = True
    tokens_per_block: int = 64
    type_id: trt.DataType
    max_context_length: int = 1024
    qkv_bias_enabled: bool = False
    do_cross_attention: bool = False
    max_distance: int = 0  # for relative attention
    pos_shift_enabled: bool = False
    dense_context_fmha: bool = False
    use_paged_context_fmha: bool = False
    use_fp8_context_fmha: bool = False
    has_full_attention_mask: bool = False
    use_cache: bool = True
    is_spec_decoding_enabled: bool = False
    spec_decoding_is_generation_length_variable: bool = False
    spec_decoding_max_generation_length: int = 1
    is_mla_enabled: bool = False
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    skip_attn: bool = False
    cp_size: int = 1
    cp_rank: int = 0
    cp_group: int = 0
    use_logn_scaling: bool = False

    def get_plugin_fields(self) -> list[trt.PluginField]:
        """Convert configuration fields to TensorRT plugin fields.

        Returns:
            List of TensorRT plugin fields with appropriate data types
        """

        def convert_to_plugin_field(name: str, value: Any) -> trt.PluginField:
            dtype: type[np.number]
            if name in ("use_cache", "mask_type", "paged_kv_cache") or (
                isinstance(value, trt.DataType | IntFlag | int) and not isinstance(value, bool | IntEnum)
            ):
                dtype = np.int32
            elif isinstance(value, float):
                dtype = np.float32
            elif isinstance(value, bool | IntEnum):
                dtype = np.int8
            else:
                raise NotImplementedError(f"Converting attribute {name} of type {type(value)} is not implemented yet")
            plugin_field_type = PLUGIN_FIELD_TYPES[dtype]
            if isinstance(value, IntEnum | IntFlag | trt.DataType):
                value = value.value
            return trt.PluginField(name, np.array(value, dtype=dtype), plugin_field_type)

        return [convert_to_plugin_field(name, value) for name, value in self.model_dump().items()]


class GPTAttentionPluginInputs(StrictlyTyped):
    """Input tensors required for GPT attention plugin.

    Contains all the input tensors needed to run the GPT attention plugin,
    including sequence information, cache indices, and optional RoPE tensors.
    """

    sequence_length: Node
    host_past_key_value_lengths: Node
    host_max_attention_window_sizes: Node
    host_sink_token_length: Node
    context_lengths: Node
    cache_indirection: Node
    host_request_types: Node
    kv_cache_block_offsets: Node
    host_kv_cache_block_offsets: Node
    host_kv_cache_pool_pointers: Node
    host_kv_cache_pool_mapping: Node
    rotary_inv_freq: Node | None = None
    rotary_cos_sin: Node | None = None
    host_context_lengths: Node
    host_runtime_perf_knobs: Node
    host_context_progress: Node

    @classmethod
    def find_from(cls, graph: Graph, is_rope: bool) -> Self:
        """Find plugin input nodes from a graph.

        Args:
            graph: FX graph to search for input nodes
            is_rope: Whether RoPE inputs should be included

        Returns:
            GPTAttentionPluginInputs instance with found nodes
        """
        existing_placeholders = {p.name: p for p in graph.find_nodes(op="placeholder")}
        get_attr_nodes = {n.name: n for n in graph.nodes if n.op == "get_attr"}
        excluded = set() if is_rope else {"rotary_inv_freq", "rotary_cos_sin"}
        nodes = {
            name: node
            # pylint: disable-next=bad-reversed-sequence
            for name in reversed(cls.model_fields)
            if (
                name not in excluded
                and isinstance(
                    node := existing_placeholders.get(name, get_attr_nodes.get(name, None)),
                    Node,
                )
            )
        }
        return cls(**nodes)


class GPTAttentionPlugin(GPTAttentionPluginFields):
    """TensorRT plugin implementation of GPT attention.

    Implements the GPT attention mechanism as a TensorRT plugin for optimized performance.
    """

    @property
    def __name__(self) -> str:
        return "gpt_attention_plugin"

    def __hash__(self) -> int:
        return hash(f"gpt_attention_plugin_{self.layer_idx}")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GPTAttentionPlugin):
            return self is other
        return False

    def __call__(
        self,
        qkv: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply GPT attention to input tensor.

        Args:
            qkv: Combined query, key and value tensor
            **kwargs: Additional input tensors

        Returns:
            Output tensor after applying attention
        """
        if is_in_fake_tensor_mode():
            # Note that this is merely for the fake tensor propagation
            q_size = self.num_heads * self.head_size
            return qkv[..., :q_size]
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")
