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

from typing import Any

import numpy as np
import tensorrt as trt
import torch
from pydantic import Field, PrivateAttr
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

from ...constants import DEFAULT_ROTARY_EMBEDDING_ORIGINAL_MAX_POSITIONS
from ...debug import open_debug_artifact
from ...types import StrictlyTyped
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin
from .utils import lookup_attributes


class Llama3ScalingConfig(StrictlyTyped):
    """Configuration for Llama 3's RoPE scaling parameters.

    Contains scaling factors for position embeddings used in Llama 3's attention mechanism.

    Attributes:
        factor (float): Main scaling factor for position embeddings. Defaults to 8.0.
        low_freq_factor (float): Scaling factor for low frequency components. Defaults to 1.0.
        high_freq_factor (float): Scaling factor for high frequency components. Defaults to 4.0.
        original_max_position_embeddings (int): Original maximum sequence length. Defaults to 8192.
    """

    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


# pylint: disable-next=too-many-instance-attributes
class ROPEConfig(StrictlyTyped):
    """Configuration for RoPE (Rotary Position Embedding) parameters.

    Contains parameters for RoPE used in the attention mechanism.

    Attributes:
        position_embedding_type (PositionEmbeddingType): Type of position embedding. Defaults to learned_absolute.
        rotary_embedding_dim (int): Dimension of rotary embeddings. Defaults to 0.
        rotary_embedding_base (float): Base value for rotary embeddings. Defaults to 10000.0.
        rotary_embedding_scale_type (RotaryScalingType): Type of scaling for rotary embeddings. Defaults to none.
        rotary_embedding_scale (float): Scale factor for rotary embeddings. Defaults to 1.0.
        rotary_embedding_short_m_scale (float): Scale for short-range rotary embeddings. Defaults to 1.0.
        rotary_embedding_long_m_scale (float): Scale for long-range rotary embeddings. Defaults to 1.0.
        rotary_embedding_max_positions (int): Maximum positions for rotary embeddings. Defaults to 1024.
        rotary_embedding_original_max_positions (int): Original maximum positions. Defaults to 1024.
        llama3_scaling_config (Llama3ScalingConfig): Configuration for Llama3 specific scaling.
            Defaults to Llama3ScalingConfig().
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
    rotary_embedding_beta_fast: int | None = Field(default=None, exclude=True)
    rotary_embedding_beta_slow: int | None = Field(default=None, exclude=True)
    rotary_embedding_mscale: float | None = Field(default=None, exclude=True)
    rotary_embedding_mscale_all_dim: float | None = Field(default=None, exclude=True)
    _longrope_scaling_short_factors: list[float] = PrivateAttr(default_factory=list)
    _longrope_scaling_long_factors: list[float] = PrivateAttr(default_factory=list)
    _rotary_inv_freq: np.ndarray | None = PrivateAttr(default=None)
    _rotary_cos_sin: np.ndarray = PrivateAttr(default_factory=lambda: np.array([]))
    _long_rope_rotary_inv_freq: np.ndarray | None = PrivateAttr(default=None)
    _long_rope_rotary_cos_sin: np.ndarray | None = PrivateAttr(default=None)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ROPEConfig):
            return False
        return self.model_dump() == other.model_dump()

    @property
    def is_rope(self) -> bool:
        """Check if the RoPE configuration is valid.

        Returns:
            bool: True if using RoPE embeddings and dimension is non-zero
        """
        return (self.position_embedding_type.is_rope() and self.rotary_embedding_dim != 0) or (
            self.rotary_embedding_scale_type == RotaryScalingType.yarn
        )

    @property
    def longrope_scaling_short_factors(self) -> np.ndarray:
        """Get the short-range scaling factors for long RoPE.

        This attribute is not a direct field for GPTAttentionPlugin, but it is used to calculate RoPE constants
        for LongRoPE.

        Returns:
            np.ndarray: Array of float32 scaling factors for short-range components.
        """
        return np.asarray(self._longrope_scaling_short_factors).astype(np.float32)

    @longrope_scaling_short_factors.setter
    def longrope_scaling_short_factors(self, value: list[float]) -> None:
        """Set the short-range scaling factors for long RoPE.

        Args:
            value (list[float]): List of scaling factors for short-range components.
        """
        self._longrope_scaling_short_factors = value

    @property
    def longrope_scaling_long_factors(self) -> np.ndarray:
        """Get the long-range scaling factors for long RoPE.

        This attribute is not a direct field for GPTAttentionPlugin, but it is used to calculate RoPE constants
        for LongRoPE.

        Returns:
            np.ndarray: Array of float32 scaling factors for long-range components.
        """
        return np.asarray(self._longrope_scaling_long_factors).astype(np.float32)

    @longrope_scaling_long_factors.setter
    def longrope_scaling_long_factors(self, value: list[float]) -> None:
        """Set the long-range scaling factors for long RoPE.

        Args:
            value (list[float]): List of scaling factors for long-range components.
        """
        self._longrope_scaling_long_factors = value

    @property
    def rotary_inv_freq(self) -> torch.nn.Parameter | None:
        """Get the inverse frequency tensor for rotary embeddings.

        This attribute is used as an input for GPTAttentionPlugin.

        Returns:
            torch.nn.Parameter: Inverse frequency tensor as a Parameter.
        """
        if self._rotary_inv_freq is None:
            return None
        return torch.nn.Parameter(torch.from_numpy(self._rotary_inv_freq))

    @rotary_inv_freq.setter
    def rotary_inv_freq(self, value: np.ndarray | None) -> None:
        """Set the inverse frequency tensor for rotary embeddings.

        Args:
            value (np.ndarray | None): Numpy array containing inverse frequencies,
                or None to disable.
        """
        self._rotary_inv_freq = value

    @property
    def rotary_cos_sin(self) -> torch.nn.Parameter:
        """Get the precomputed cosine/sine tensor for rotary embeddings.

        This attribute is used as an input for GPTAttentionPlugin.

        Returns:
            torch.nn.Parameter: Cosine/sine tensor as a Parameter.
        """
        return torch.nn.Parameter(torch.from_numpy(self._rotary_cos_sin))

    @rotary_cos_sin.setter
    def rotary_cos_sin(self, value: np.ndarray) -> None:
        """Set the precomputed cosine/sine tensor for rotary embeddings.

        Args:
            value (np.ndarray): Numpy array containing cosine/sine values.
        """
        self._rotary_cos_sin = value

    @property
    def long_rope_rotary_inv_freq(self) -> torch.nn.Parameter | None:
        """Get the inverse frequency tensor for long-range rotary embeddings.

        This attribute is used as an input for GPTAttentionPlugin when using LongRoPE.

        Returns:
            torch.nn.Parameter | None: Inverse frequency tensor as a Parameter if available,
                None otherwise.
        """
        if self._long_rope_rotary_inv_freq is None:
            return None
        return torch.nn.Parameter(torch.from_numpy(self._long_rope_rotary_inv_freq))

    @long_rope_rotary_inv_freq.setter
    def long_rope_rotary_inv_freq(self, value: np.ndarray | None) -> None:
        """Set the inverse frequency tensor for long-range rotary embeddings.

        Args:
            value (np.ndarray | None): Numpy array containing inverse frequencies,
                or None to disable.
        """
        self._long_rope_rotary_inv_freq = value

    @property
    def long_rope_rotary_cos_sin(self) -> torch.nn.Parameter | None:
        """Get the precomputed cosine/sine tensor for long-range rotary embeddings.

        This attribute is used as an input for GPTAttentionPlugin when using LongRoPE.

        Returns:
            torch.nn.Parameter | None: Cosine/sine tensor as a Parameter if available,
                None otherwise.
        """
        if self._long_rope_rotary_cos_sin is None:
            return None
        return torch.nn.Parameter(torch.from_numpy(self._long_rope_rotary_cos_sin))

    @long_rope_rotary_cos_sin.setter
    def long_rope_rotary_cos_sin(self, value: np.ndarray | None) -> None:
        """Set the precomputed cosine/sine tensor for long-range rotary embeddings.

        Args:
            value (np.ndarray | None): Numpy array containing cosine/sine values,
                or None to disable.
        """
        self._long_rope_rotary_cos_sin = value

    @classmethod
    def from_pretrained_config(
        cls,
        pretrained_config: PretrainedConfig | None = None,
        positional_embedding_type: PositionEmbeddingType | None = None,
        rotary_embedding_dim: int | None = None,
    ) -> Self:
        """Create a RoPE configuration from a HuggingFace pretrained model configuration.

        This method extracts RoPE (Rotary Position Embedding) parameters from a HuggingFace
        pretrained model configuration and creates a corresponding ROPEConfig object.

        Args:
            pretrained_config (PretrainedConfig | None): HuggingFace pretrained model configuration object.
                If None, default values will be used. Defaults to None.
            positional_embedding_type (PositionEmbeddingType | None): Override the position embedding type.
                If None, uses the type from pretrained_config. Defaults to None.
            rotary_embedding_dim (int | None): Override the rotary embedding dimension.
                If None, uses the dimension from pretrained_config. Defaults to None.

        Returns:
            Self: A ROPEConfig object initialized with parameters from the pretrained config,
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
        model_type = lookup_attributes(
            pretrained_config,
            "model_type",
            default="unknown",
        )
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

        if positional_embedding_type is not None:
            rope_config.position_embedding_type = positional_embedding_type
        if rotary_embedding_dim is not None:
            rope_config.rotary_embedding_dim = rotary_embedding_dim
        if rope_scaling:
            rope_config.rotary_embedding_scale = rope_scaling.get("factor", rope_config.rotary_embedding_scale)
            rope_config.rotary_embedding_original_max_positions = rope_scaling.get(
                "original_max_position_embeddings", DEFAULT_ROTARY_EMBEDDING_ORIGINAL_MAX_POSITIONS
            )
            rotary_scaling_type: str | None = rope_scaling.get("type", rope_scaling.get("rope_type", None))
            if rotary_scaling_type is not None:
                rope_config.rotary_embedding_scale_type = RotaryScalingType.from_string(rotary_scaling_type)
            rope_config.llama3_scaling_config = Llama3ScalingConfig(**rope_scaling)
            if rotary_scaling_type == "longrope":
                assert all(factor in rope_scaling for factor in ("short_factor", "long_factor"))
                # NOTE: TensorRT-LLM uses long_rope position_embedding_type for phi-3.5,
                #       but setting it to long_rope results in an error in Ditto.
                #       However, since this doesn't affect functionality, we can ignore it for now.
                # rope_config.position_embedding_type = PositionEmbeddingType.long_rope
                # TODO: Fix above issue.
                original_max_position_embeddings = lookup_attributes(
                    pretrained_config,
                    "original_max_position_embeddings",
                    default=rope_config.rotary_embedding_original_max_positions,
                )
                rope_config.rotary_embedding_original_max_positions = original_max_position_embeddings
                rope_config.longrope_scaling_short_factors = rope_scaling["short_factor"]
                rope_config.longrope_scaling_long_factors = rope_scaling["long_factor"]
            if model_type == "deepseek_v2":
                rope_config.position_embedding_type = PositionEmbeddingType.learned_absolute
                rope_config.rotary_embedding_beta_fast = rope_scaling.get("beta_fast", None)
                rope_config.rotary_embedding_beta_slow = rope_scaling.get("beta_slow", None)
                rope_config.rotary_embedding_mscale = rope_scaling.get("mscale", None)
                rope_config.rotary_embedding_mscale_all_dim = rope_scaling.get("mscale_all_dim", None)
                rope_config.rotary_embedding_dim = 0

        return rope_config

    def compute_rope_constants(self, qk_rope_head_dim: int | None = None) -> None:
        """Compute RoPE constants used by the GPT attention plugin.

        This method computes the inverse frequency and cosine/sine tensors needed for RoPE,
        handling both standard RoPE and long RoPE configurations.
        """
        # TODO: replace this by `Attention.create_attention_const_params`
        rotary_inv_freq, long_rope_rotary_inv_freq, long_rope_embed_positions_for_gpt_attention = None, None, None
        match self.rotary_embedding_scale_type:
            case RotaryScalingType.longrope:
                (
                    _,
                    _,
                    (rotary_inv_freq, embed_positions_for_gpt_attention),
                    (long_rope_rotary_inv_freq, long_rope_embed_positions_for_gpt_attention),
                    mscale,
                ) = RopeEmbeddingUtils.create_sinusoidal_positions_long_rope(
                    self.rotary_embedding_max_positions,
                    self.rotary_embedding_original_max_positions,
                    self.rotary_embedding_dim,
                    self.rotary_embedding_base,
                    self.longrope_scaling_short_factors,
                    self.longrope_scaling_long_factors,
                )
                self.rotary_embedding_short_m_scale = self.rotary_embedding_long_m_scale = mscale

            case RotaryScalingType.yarn:
                assert self.rotary_embedding_beta_fast is not None
                assert self.rotary_embedding_beta_slow is not None
                assert self.rotary_embedding_mscale is not None
                assert self.rotary_embedding_mscale_all_dim is not None
                assert qk_rope_head_dim is not None
                embed_positions_for_gpt_attention = (
                    RopeEmbeddingUtils.create_sinusoidal_positions_for_deepseek_attention_plugin(
                        self.rotary_embedding_max_positions,
                        qk_rope_head_dim,
                        int(self.rotary_embedding_base),
                        self.rotary_embedding_scale,
                        self.rotary_embedding_original_max_positions,
                        self.rotary_embedding_beta_fast,
                        self.rotary_embedding_beta_slow,
                        self.rotary_embedding_mscale,
                        self.rotary_embedding_mscale_all_dim,
                    )
                )

            case _:
                (
                    rotary_inv_freq,
                    embed_positions_for_gpt_attention,
                ) = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                    self.rotary_embedding_max_positions,
                    self.rotary_embedding_dim,
                    self.rotary_embedding_base,
                    self.rotary_embedding_scale,
                    self.rotary_embedding_scale_type,
                    self.llama3_scaling_config.model_dump(),
                )

        self.rotary_inv_freq = rotary_inv_freq
        self.rotary_cos_sin = embed_positions_for_gpt_attention
        self.long_rope_rotary_inv_freq = long_rope_rotary_inv_freq
        self.long_rope_rotary_cos_sin = long_rope_embed_positions_for_gpt_attention

    def save_debug_artifacts(self) -> None:
        """Save debug artifacts for RoPE configuration and inputs."""
        with open_debug_artifact("rope_inputs.pt", "wb") as f:
            if f:
                rope_inputs = {
                    "rotary_cos_sin": self.rotary_cos_sin,
                }
                if self.rotary_inv_freq is not None:
                    rope_inputs.update(
                        {
                            "rotary_inv_freq": self.rotary_inv_freq,
                        }
                    )
                if self.long_rope_rotary_inv_freq is not None and self.long_rope_rotary_cos_sin is not None:
                    rope_inputs.update(
                        {
                            "long_rope_rotary_inv_freq": self.long_rope_rotary_inv_freq,
                            "long_rope_rotary_cos_sin": self.long_rope_rotary_cos_sin,
                        }
                    )
                torch.save(
                    rope_inputs,
                    f,
                )
        with open_debug_artifact("rope_config.json") as f:
            if f:
                f.write(self.model_dump_json(indent=2))


class GPTAttentionPlugin(Plugin):
    """TensorRT plugin for GPT attention mechanism.

    Attributes:
        layer_idx (int): Index of the attention layer
        num_heads (int): Number of attention heads
        vision_start (int): Starting position for vision tokens. Defaults to -1.
        vision_length (int): Length of vision tokens. Defaults to -1.
        num_kv_heads (int): Number of key/value heads
        layer_idx_in_cache_pool (int): Layer index in the KV cache pool
        head_size (int): Size of each attention head
        unidirectional (int): Whether attention is unidirectional. Defaults to 1.
        q_scaling (float): Query scaling factor. Defaults to 1.0.
        attn_logit_softcapping_scale (float): Scale for attention logit soft capping. Defaults to 0.0.
        position_embedding_type (PositionEmbeddingType): Type of position embedding. Defaults to learned_absolute.
        rotary_embedding_dim (int): Dimension of rotary embeddings. Defaults to 0.
        rotary_embedding_base (float): Base value for rotary embeddings. Defaults to 10000.0.
        rotary_embedding_scale_type (RotaryScalingType): Type of scaling for rotary embeddings. Defaults to none.
        rotary_embedding_scale (float): Scale factor for rotary embeddings. Defaults to 1.0.
        rotary_embedding_short_m_scale (float): Scale for short-range rotary embeddings. Defaults to 1.0.
        rotary_embedding_long_m_scale (float): Scale for long-range rotary embeddings. Defaults to 1.0.
        rotary_embedding_max_positions (int): Maximum positions for rotary embeddings. Defaults to 1024.
        rotary_embedding_original_max_positions (int): Original maximum positions. Defaults to 1024.
        tp_size (int): Tensor parallel size. Defaults to 1.
        tp_rank (int): Tensor parallel rank. Defaults to 0.
        unfuse_qkv_gemm (bool): Whether to unfuse QKV GEMM. Defaults to False.
        context_fmha_type (ContextFMHAType): Type of context FMHA. Defaults to enabled.
        kv_cache_quant_mode (QuantMode): Quantization mode for KV cache. Defaults to QuantMode(0).
        remove_input_padding (bool): Whether to remove input padding. Defaults to True.
        mask_type (AttentionMaskType): Type of attention mask. Defaults to causal.
        block_sparse_block_size (int): Block size for sparse attention. Defaults to 64.
        block_sparse_homo_head_pattern (bool): Whether to use homogeneous head pattern. Defaults to False.
        block_sparse_num_local_blocks (int): Number of local blocks. Defaults to 16.
        block_sparse_vertical_stride (int): Vertical stride for sparse blocks. Defaults to 8.
        paged_kv_cache (bool): Whether to use paged KV cache. Defaults to True.
        tokens_per_block (int): Number of tokens per block. Defaults to 64.
        type_id (trt.DataType): Data type for computation
        max_context_length (int): Maximum context length. Defaults to 1024.
        qkv_bias_enabled (bool): Whether QKV bias is enabled. Defaults to False.
        do_cross_attention (bool): Whether to do cross attention. Defaults to False.
        max_distance (int): Maximum distance for relative attention. Defaults to 0.
        pos_shift_enabled (bool): Whether position shift is enabled. Defaults to False.
        dense_context_fmha (bool): Whether to use dense context FMHA. Defaults to False.
        use_paged_context_fmha (bool): Whether to use paged context FMHA. Defaults to False.
        use_fp8_context_fmha (bool): Whether to use FP8 context FMHA. Defaults to False.
        has_full_attention_mask (bool): Whether full attention mask is used. Defaults to False.
        use_cache (bool): Whether to use KV cache. Defaults to True.
        is_spec_decoding_enabled (bool): Whether speculative decoding is enabled. Defaults to False.
        spec_decoding_is_generation_length_variable (bool): Whether generation length is variable. Defaults to False.
        spec_decoding_max_generation_length (int): Maximum generation length. Defaults to 1.
        is_mla_enabled (bool): Whether MLA is enabled. Defaults to False.
        q_lora_rank (int): Rank for Q LoRA. Defaults to 0.
        kv_lora_rank (int): Rank for KV LoRA. Defaults to 0.
        qk_nope_head_dim (int): Head dimension for QK without position embeddings. Defaults to 0.
        qk_rope_head_dim (int): Head dimension for QK with RoPE. Defaults to 0.
        v_head_dim (int): Head dimension for value. Defaults to 0.
        skip_attn (bool): Whether to skip attention. Defaults to False.
        cp_size (int): Checkpoint size. Defaults to 1.
        cp_rank (int): Checkpoint rank. Defaults to 0.
        cp_group (int): Checkpoint group. Defaults to 0.
        use_logn_scaling (bool): Whether to use log(n) attention scaling. Defaults to False.
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

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for a plugin field value.

        Args:
            name (str): Name of the field
            value (Any): Value to get dtype for

        Returns:
            type[np.number]: numpy dtype for the value
        """
        if name in ("use_cache", "mask_type", "paged_kv_cache"):
            return np.int32
        return super().get_field_dtype(name, value)

    def __call__(
        self,
        qkv: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply attention to input tensor.

        Args:
            qkv (torch.Tensor): Input tensor containing query, key and value
            **kwargs (Any): Additional keyword arguments

        Returns:
            torch.Tensor: Output tensor after applying attention

        Raises:
            NotImplementedError: If not in fake tensor mode
        """
        if is_in_fake_tensor_mode():
            # Note that this is merely for the fake tensor propagation
            head_size = self.head_size if self.v_head_dim == 0 else self.v_head_dim
            q_size = self.num_heads * head_size
            return qkv[..., :q_size]
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")


class GPTAttentionPluginInputs(StrictlyTyped):
    """Input tensors required for GPT attention plugin.

    Contains all the input tensors needed to run the GPT attention plugin,
    including sequence information, cache indices, and optional RoPE tensors.

    Attributes:
        sequence_length (Node): Length of input sequence
        host_past_key_value_lengths (Node): Past key/value lengths on host
        host_max_attention_window_sizes (Node): Maximum attention window sizes
        host_sink_token_length (Node): Length of sink tokens
        context_lengths (Node): Context lengths
        cache_indirection (Node): Cache indirection tensor
        host_request_types (Node): Request types on host
        kv_cache_block_offsets (Node): KV cache block offsets
        host_kv_cache_block_offsets (Node): KV cache block offsets on host
        host_kv_cache_pool_pointers (Node): KV cache pool pointers on host
        host_kv_cache_pool_mapping (Node): KV cache pool mapping on host
        rotary_inv_freq (Node | None): Inverse frequency tensor for RoPE. Defaults to None.
        rotary_cos_sin (Node | None): Cosine/sine tensor for RoPE. Defaults to None.
        host_context_lengths (Node): Context lengths on host
        host_runtime_perf_knobs (Node): Runtime performance knobs on host
        host_context_progress (Node): Context progress on host
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
    kv_orig_quant_scale: Node | None = None
    kv_quant_orig_scale: Node | None = None
    rotary_inv_freq: Node | None = None
    rotary_cos_sin: Node | None = None
    host_context_lengths: Node
    host_runtime_perf_knobs: Node
    host_context_progress: Node

    @classmethod
    def find_from(cls, graph: Graph, is_rope: bool) -> Self:
        """Find plugin input nodes from a graph.

        Args:
            graph (Graph): FX graph to search for input nodes
            is_rope (bool): Whether RoPE inputs should be included

        Returns:
            Self: GPTAttentionPluginInputs instance with found nodes
        """
        existing_placeholders = {p.name: p for p in graph.find_nodes(op="placeholder")}
        get_attr_nodes = {n.name: n for n in graph.nodes if n.op == "get_attr"}
        excluded: set[str] = {"kv_orig_quant_scale", "kv_quant_orig_scale"}
        if not is_rope:
            excluded.update({"rotary_inv_freq", "rotary_cos_sin"})
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
