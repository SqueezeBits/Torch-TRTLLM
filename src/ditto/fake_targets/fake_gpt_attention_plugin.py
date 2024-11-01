import logging
from enum import IntEnum, IntFlag
from typing import Any, TypeVar

import numpy as np
import tensorrt as trt
import torch
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

from ..types import StrictlyTyped

logger = logging.getLogger(__name__)


class Llama3ScalingConfig(StrictlyTyped):
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


class ROPEConfig(StrictlyTyped):
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

    @classmethod
    def from_pretrained_config(
        cls,
        pretrained_config: PretrainedConfig | None = None,
        positional_embedding_type: PositionEmbeddingType | None = None,
        embedding_dim: int | None = None,
    ) -> Self:
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
        if positional_embedding_type is not None:
            rope_config.position_embedding_type = positional_embedding_type
        if embedding_dim is not None:
            rope_config.rotary_embedding_dim = embedding_dim
        return rope_config

    def compute_rope_constants(self) -> tuple[np.ndarray, np.ndarray]:
        # TODO: replace this by `Attention.create_attention_const_params`
        rotary_inv_freq, embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            self.rotary_embedding_max_positions,
            self.rotary_embedding_dim,
            self.rotary_embedding_base,
            self.rotary_embedding_scale,
            self.rotary_embedding_scale_type,
            self.llama3_scaling_config.model_dump(),
        )
        return rotary_inv_freq, embed_positions


T = TypeVar("T")


def lookup_attributes(pretrained_config: PretrainedConfig | None, *names: str, default: T) -> T:
    if pretrained_config is None:
        return default
    for name in names:
        if hasattr(pretrained_config, name):
            return getattr(pretrained_config, name)
    logger.warning(
        "None of the following attributes are found in pretrained config. "
        f"Will use the default value {default}: {', '.join(names)}"
    )
    return default


class GPTAttentionPluginFields(StrictlyTyped):
    # the order of the attributes does matter!
    layer_idx: int
    num_heads: int
    vision_start: int = -1
    vision_length: int = -1
    num_kv_heads: int
    head_size: int  # this field is actually `hidden_size_per_head`
    unidirectional: int = 1
    q_scaling: float = 1.0
    qk_tanh_scale: float = 0.0
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
    enable_xqa: bool = True
    kv_cache_quant_mode: QuantMode = QuantMode(0)
    remove_input_padding: bool = True
    mask_type: AttentionMaskType = AttentionMaskType.causal
    block_sparse_block_size: int = 64
    block_sparse_homo_head_pattern: bool = False
    block_sparse_num_local_blocks: int = 16
    block_sparse_vertical_stride: int = 8
    paged_kv_cache: bool = True
    tokens_per_block: int = 64
    type_id: trt.DataType = trt.float16
    max_context_length: int = 1024
    qkv_bias_enabled: bool = False
    do_cross_attention: bool = False
    max_distance: int = 0  # for relative attention
    pos_shift_enabled: bool = False
    dense_context_fmha: bool = False
    use_paged_context_fmha: bool = False
    use_fp8_context_fmha: bool = False
    use_cache: bool = True
    is_spec_decoding_enabled: bool = False
    spec_decoding_is_generation_length_variable: bool = False
    spec_decoding_max_generation_length: int = 1

    def get_plugin_fields(self) -> list[trt.PluginField]:
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
            plugin_field_type = {
                np.int8: trt.PluginFieldType.INT8,
                np.int16: trt.PluginFieldType.INT16,
                np.int32: trt.PluginFieldType.INT32,
                np.float16: trt.PluginFieldType.FLOAT16,
                np.float32: trt.PluginFieldType.FLOAT32,
                np.float64: trt.PluginFieldType.FLOAT64,
            }[dtype]
            if isinstance(value, IntEnum | IntFlag | trt.DataType):
                value = value.value
            return trt.PluginField(name, np.array(value, dtype=dtype), plugin_field_type)

        return [convert_to_plugin_field(name, value) for name, value in self.model_dump().items()]


class GPTAttentionPluginInputs(StrictlyTyped):
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
    rotary_inv_freq: Node
    rotary_cos_sin: Node
    host_context_lengths: Node
    host_runtime_perf_knobs: Node

    @classmethod
    def find_from(cls, graph: Graph) -> Self:
        existing_placeholders = {p.name: p for p in graph.nodes if p.op == "placeholder"}
        get_attr_nodes = {n.name: n for n in graph.nodes if n.op == "get_attr"}
        nodes = {
            name: node
            for name in reversed(cls.model_fields)
            if isinstance(node := existing_placeholders.get(name, get_attr_nodes.get(name, None)), Node)
        }
        return cls(**nodes)


class FakeGPTAttentionPlugin(GPTAttentionPluginFields):
    @property
    def __name__(self) -> str:
        return "fake_gpt_attention_plugin"

    def __hash__(self) -> int:
        return hash(f"fake_gpt_attention_plugin_{self.layer_idx}")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeGPTAttentionPlugin):
            return self is other
        return False

    def __call__(
        self,
        qkv: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")
