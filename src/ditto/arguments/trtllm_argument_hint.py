# mypy: disable-error-code="misc"
# pylint: disable=too-many-public-methods
from collections.abc import Callable

import torch
from pydantic import Field, TypeAdapter, computed_field, model_serializer
from typing_extensions import Self

from ..configs import TRTLLMOptimizationProfileConfig
from ..constants import INPUT_IDS_UNSQUEEZE_DIM
from ..types import StrictlyTyped
from .dynamic_dim import DynamicDimension, DynamicDimensionType
from .tensor_type_hint import TensorTypeHint


class TRTLLMArgumentHint(StrictlyTyped):
    """Configuration for TensorRT-LLM model input arguments.

    Args:
        batch_size (DynamicDimensionType): Batch size dimension
        num_tokens (DynamicDimensionType): Number of tokens dimension
        kv_cache_block_size (DynamicDimensionType): KV cache block size dimension
        beam_width (DynamicDimensionType | int): Beam width dimension or fixed value
        attention_window_size (DynamicDimensionType): Attention window size dimension
        num_attn_layers (int | None): Number of attention layers. Defaults to None.
        tp_size (int): Tensor parallel size. Defaults to 1.
        lora_input_hints (dict[str, TensorTypeHint]): LoRA input tensor hints. Defaults to empty dict.
    """

    batch_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_tokens: DynamicDimensionType = Field(frozen=True, exclude=True)
    kv_cache_block_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    beam_width: DynamicDimensionType | int = Field(frozen=True, exclude=True)
    attention_window_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_attn_layers: int | None = Field(default=None, exclude=True, ge=0)
    tp_size: int = Field(default=1, exclude=True, gt=0)
    lora_input_hints: dict[str, TensorTypeHint] = Field(default_factory=dict, exclude=True)

    @classmethod
    def configure(cls, profile_config: TRTLLMOptimizationProfileConfig, *, tp_size: int = 1) -> Self:
        """Create argument hints from optimization profile config.

        Args:
            profile_config (TRTLLMOptimizationProfileConfig): Optimization profile configuration
            tp_size (int, optional): Tensor parallel size. Defaults to 1.

        Returns:
            Self: Configured argument hints
        """
        batch_size = DynamicDimension(
            name="batch_size",
            min=1,
            opt=profile_config.opt_batch_size,
            max=profile_config.max_batch_size,
        )
        ops_s = profile_config.opt_num_tokens // 8
        max_s = profile_config.max_num_tokens // 8
        s = DynamicDimension(name="s", min=0, opt=ops_s, max=max_s)
        num_tokens = 8 * s
        kv_cache_block_size = DynamicDimension(
            name="kv_cache_block_size",
            min=1,
            opt=profile_config.opt_kv_cache_block_size,
            max=profile_config.max_kv_cache_block_size,
        )
        beam_width: DynamicDimensionType | int = (
            1
            if profile_config.max_beam_width == 1
            else DynamicDimension(
                name="beam_width",
                min=1,
                opt=profile_config.opt_beam_width,
                max=profile_config.max_beam_width,
            )
        )
        attention_window_size = DynamicDimension(
            name="attention_window_size",
            min=1,
            opt=profile_config.opt_attention_window_size,
            max=profile_config.max_attention_window_size,
        )
        return cls(
            batch_size=batch_size,
            num_tokens=num_tokens,
            kv_cache_block_size=kv_cache_block_size,
            beam_width=beam_width,
            attention_window_size=attention_window_size,
            tp_size=tp_size,
        )

    def as_dict(self) -> dict[str, TensorTypeHint | None]:
        """Convert argument hints to dictionary.

        Returns:
            dict[str, TensorTypeHint | None]: Dictionary of tensor hints
        """
        return TypeAdapter(dict[str, TensorTypeHint | None]).validate_python(self.model_dump())

    @property
    def batched_input_ids(self) -> TensorTypeHint:
        """Tensor type hint for batched input IDs with shape (1, num_tokens) or (num_tokens, 1)."""
        if INPUT_IDS_UNSQUEEZE_DIM == 0:
            return TensorTypeHint(shape=(1, self.num_tokens), dtype=torch.int32)
        return TensorTypeHint(shape=(self.num_tokens, 1), dtype=torch.int32)

    @computed_field
    @property
    def input_ids(self) -> TensorTypeHint:
        """Tensor type hint for input IDs with shape (num_tokens,)."""
        return TensorTypeHint(shape=(self.num_tokens,), dtype=torch.int32)

    @computed_field
    @property
    def position_ids(self) -> TensorTypeHint:
        """Tensor type hint for position IDs with shape (num_tokens,)."""
        return TensorTypeHint(shape=(self.num_tokens,), dtype=torch.int32)

    @computed_field
    @property
    def last_token_ids(self) -> TensorTypeHint:
        """Tensor type hint for last token IDs with shape (batch_size,)."""
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def kv_cache_block_offsets(self) -> TensorTypeHint:
        """Tensor type hint for KV cache block offsets with shape (1, batch_size, 2, kv_cache_block_size)."""
        return TensorTypeHint(shape=(1, self.batch_size, 2, self.kv_cache_block_size), dtype=torch.int32)

    @computed_field
    @property
    def host_kv_cache_block_offsets(self) -> TensorTypeHint:
        """Tensor type hint for host KV cache block offsets with shape (1, batch_size, 2, kv_cache_block_size)."""
        return TensorTypeHint(shape=(1, self.batch_size, 2, self.kv_cache_block_size), dtype=torch.int32)

    @computed_field
    @property
    def host_kv_cache_pool_pointers(self) -> TensorTypeHint:
        """Tensor type hint for host KV cache pool pointers with shape (1, 2)."""
        return TensorTypeHint(shape=(1, 2), dtype=torch.int64)

    @computed_field
    @property
    def sequence_length(self) -> TensorTypeHint:
        """Tensor type hint for sequence length with shape (batch_size,)."""
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def host_request_types(self) -> TensorTypeHint:
        """Tensor type hint for host request types with shape (batch_size,)."""
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def host_past_key_value_lengths(self) -> TensorTypeHint:
        """Tensor type hint for host past key value lengths with shape (batch_size,)."""
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def context_lengths(self) -> TensorTypeHint:
        """Tensor type hint for context lengths with shape (batch_size,)."""
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def host_runtime_perf_knobs(self) -> TensorTypeHint:
        """Tensor type hint for host runtime performance knobs with shape (16,)."""
        return TensorTypeHint(shape=(16,), dtype=torch.int64)

    @computed_field
    @property
    def host_context_lengths(self) -> TensorTypeHint:
        """Tensor type hint for host context lengths with shape (batch_size,)."""
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def host_max_attention_window_sizes(self) -> TensorTypeHint:
        """Tensor type hint for host max attention window sizes with shape (num_attn_layers,)."""
        assert self.num_attn_layers is not None, "num_attn_layers needs to be set for host_max_attention_window_sizes"
        return TensorTypeHint(shape=(self.num_attn_layers,), dtype=torch.int32)

    @computed_field
    @property
    def host_sink_token_length(self) -> TensorTypeHint:
        """Tensor type hint for host sink token length with shape (1,)."""
        return TensorTypeHint(shape=(1,), dtype=torch.int32)

    @computed_field
    @property
    def cache_indirection(self) -> TensorTypeHint:
        """Tensor type hint for cache indirection with shape (batch_size, beam_width, attention_window_size)."""
        return TensorTypeHint(
            shape=(self.batch_size, self.beam_width, self.attention_window_size),
            dtype=torch.int32,
        )

    @computed_field
    @property
    def host_kv_cache_pool_mapping(self) -> TensorTypeHint:
        """Tensor type hint for host KV cache pool mapping with shape (num_attn_layers,)."""
        assert self.num_attn_layers is not None, "num_attn_layers needs to be set for host_kv_cache_pool_mapping"
        return TensorTypeHint(shape=(self.num_attn_layers,), dtype=torch.int32)

    @computed_field
    @property
    def host_context_progress(self) -> TensorTypeHint:
        """Tensor type hint for host context progress with shape (1,)."""
        return TensorTypeHint(shape=(1,), dtype=torch.int64)

    @computed_field
    @property
    def all_reduce_workspace(self) -> TensorTypeHint | None:
        """Tensor type hint for all reduce workspace with shape (workspace_size,) or None if tp_size is 1."""
        if self.tp_size == 1:
            return None
        pointers_per_rank = 7
        pointers_of_counter = 2
        workspace_size = pointers_per_rank * self.tp_size + pointers_of_counter
        return TensorTypeHint(shape=(workspace_size,), dtype=torch.int64)

    @model_serializer(mode="wrap")
    def serialize_model(
        self,
        original_serializer: Callable[[Self], dict[str, TensorTypeHint | None]],
    ) -> dict[str, TensorTypeHint | None]:
        """Serialize model including LoRA input hints.

        Args:
            original_serializer (Callable[[Self], dict[str, TensorTypeHint | None]]): Original serializer function

        Returns:
            dict[str, TensorTypeHint | None]: Serialized model with LoRA hints
        """
        data = original_serializer(self)
        data.update(self.lora_input_hints)
        return data
