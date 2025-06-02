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

# mypy: disable-error-code="misc"
# pylint: disable=too-many-public-methods

from collections.abc import Callable

import torch
from pydantic import Field, PrivateAttr, TypeAdapter, computed_field, model_serializer
from typing_extensions import Self

from ..configs import TRTLLMMapping, TRTLLMOptimizationProfileConfig
from ..constants import INPUT_IDS_UNSQUEEZE_DIM
from ..types import StrictlyTyped
from .dynamic_dim import DynamicDimension, DynamicDimensionType
from .tensor_type_hint import TensorTypeHint


# pylint: disable=too-many-public-methods
class TRTLLMArgumentHint(StrictlyTyped):
    """Configuration for TensorRT-LLM model input arguments.

    Args:
        batch_size (DynamicDimensionType): Batch size dimension
        max_len (DynamicDimensionType): Maximum sequence length dimension
        num_tokens (DynamicDimensionType): Number of tokens dimension
        max_blocks_per_seq (DynamicDimensionType): Maximum number of blocks per sequence dimension
        beam_width (DynamicDimensionType | int): Beam width dimension or fixed value
        mapping (TRTLLMMapping): Mapping configuration
        num_attn_layers (int | None): Number of attention layers. Defaults to None.
        gather_context_logits (bool): Whether to gather context logits. Defaults to False.
        lora_input_hints (dict[str, TensorTypeHint]): LoRA input tensor hints. Defaults to empty dict.
        hidden_size (int | None): Hidden size for hidden_states_input. Defaults to None.
            (Required only when pp_rank > 0)
        hidden_dtype (torch.dtype | None): Hidden dtype for hidden_states_input. Defaults to None.
            (Required only when pp_rank > 0)
    """

    batch_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    max_len: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_tokens: DynamicDimensionType = Field(frozen=True, exclude=True)
    max_blocks_per_seq: DynamicDimensionType = Field(frozen=True, exclude=True)
    beam_width: DynamicDimensionType | int = Field(frozen=True, exclude=True)
    mapping: TRTLLMMapping = Field(exclude=True)
    num_attn_layers: int | None = Field(default=None, exclude=True, ge=0)
    gather_context_logits: bool = Field(default=False, exclude=True)
    lora_input_hints: dict[str, TensorTypeHint] = Field(default_factory=dict, exclude=True)
    hidden_size: int | None = Field(default=None, exclude=True)
    hidden_dtype: torch.dtype | None = Field(default=None, exclude=True)
    mrope_input_hints: dict[str, TensorTypeHint] = Field(default_factory=dict, exclude=True)
    _one: DynamicDimension = PrivateAttr(default=DynamicDimension(name="one", min=1, opt=1, max=1))

    @classmethod
    def configure(
        cls,
        profile_config: TRTLLMOptimizationProfileConfig,
        *,
        mapping: TRTLLMMapping,
        gather_context_logits: bool,
    ) -> Self:
        """Configure the argument hint.

        Args:
            profile_config (TRTLLMOptimizationProfileConfig): The optimization profile configuration
            mapping (TRTLLMMapping): The mapping configuration
            gather_context_logits (bool): Whether to gather context logits

        Returns:
            Self: The configured argument hint
        """
        batch_size = DynamicDimension(
            name="batch_size",
            min=1,
            opt=profile_config.opt_batch_size,
            max=profile_config.max_batch_size,
        )
        max_len = DynamicDimension(
            name="max_len",
            min=1,
            opt=profile_config.opt_seq_len,
            max=profile_config.max_seq_len,
        )
        num_tokens = DynamicDimension(
            name="num_tokens",
            min=1,
            opt=profile_config.opt_num_tokens,
            max=profile_config.max_num_tokens,
        )
        max_blocks_per_seq = DynamicDimension(
            name="max_blocks_per_seq",
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
        return cls(
            batch_size=batch_size,
            max_len=max_len,
            num_tokens=num_tokens,
            max_blocks_per_seq=max_blocks_per_seq,
            beam_width=beam_width,
            mapping=mapping,
            gather_context_logits=gather_context_logits,
        )

    def as_dict(self) -> dict[str, TensorTypeHint]:
        """Convert argument hints to dictionary.

        Returns:
            dict[str, TensorTypeHint]: Dictionary of tensor hints
        """
        return TypeAdapter(dict[str, TensorTypeHint]).validate_python(self.model_dump(exclude_none=True))

    @property
    def num_attn_layers_per_pipeline(self) -> int:
        """Number of attention layers per pipeline.

        Returns:
            int: Number of attention layers per pipeline
        """
        assert (
            self.num_attn_layers is not None
        ), "num_attn_layers needs to be set for getting num_attn_layers_per_pipeline"
        return self.num_attn_layers // self.mapping.pp_size

    @property
    def batched_input_ids(self) -> TensorTypeHint:
        """Tensor type hint for batched input IDs with shape (1, num_tokens) or (num_tokens, 1)."""
        if INPUT_IDS_UNSQUEEZE_DIM == 0:
            return TensorTypeHint(shape=(1, self.num_tokens), dtype=torch.int32)
        return TensorTypeHint(shape=(self.num_tokens, 1), dtype=torch.int32)

    @computed_field
    @property
    def input_ids(self) -> TensorTypeHint | None:
        """Tensor type hint for input IDs with shape (num_tokens,).

        If the pipeline parallelism is used, the input IDs may not be needed.
        """
        return TensorTypeHint(shape=(self.num_tokens,), dtype=torch.int32) if self.mapping.is_first_pp_rank() else None

    @computed_field
    @property
    def hidden_states_input(self) -> TensorTypeHint | None:
        """Tensor type hint for hidden states input with shape (num_tokens, hidden_size).

        It is used for pipeline parallel.
        """
        return (
            TensorTypeHint(shape=(self.num_tokens, self.hidden_size), dtype=self.hidden_dtype)
            if not self.mapping.is_first_pp_rank() and self.hidden_size is not None and self.hidden_dtype is not None
            else None
        )

    @computed_field
    @property
    def position_ids(self) -> TensorTypeHint:
        """Tensor type hint for position IDs with shape (num_tokens,)."""
        return TensorTypeHint(shape=(self.num_tokens,), dtype=torch.int32)

    @computed_field
    @property
    def last_token_ids(self) -> TensorTypeHint | None:
        """Tensor type hint for last token IDs with shape (num_tokens,), None if gather context logits is True."""
        if self.gather_context_logits or not self.mapping.is_last_pp_rank():
            return None
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def kv_cache_block_offsets(self) -> TensorTypeHint:
        """Tensor type hint for KV cache block offsets with shape (1, batch_size, 2, kv_cache_block_size)."""
        return TensorTypeHint(shape=(1, self.batch_size, 2, self.max_blocks_per_seq), dtype=torch.int32)

    @computed_field
    @property
    def host_kv_cache_block_offsets(self) -> TensorTypeHint:
        """Tensor type hint for host KV cache block offsets with shape (1, batch_size, 2, kv_cache_block_size)."""
        return TensorTypeHint(shape=(1, self.batch_size, 2, self.max_blocks_per_seq), dtype=torch.int32)

    @computed_field
    @property
    def host_kv_cache_pool_pointers(self) -> TensorTypeHint:
        """Tensor type hint for host KV cache pool pointers with shape (1, 2)."""
        return TensorTypeHint(shape=(self._one, 2), dtype=torch.int64)

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
        return TensorTypeHint(
            shape=(
                DynamicDimension(
                    name="host_runtime_perf_knobs",
                    min=16,
                    opt=16,
                    max=16,
                ),
            ),
            dtype=torch.int64,
        )

    @computed_field
    @property
    def host_context_lengths(self) -> TensorTypeHint:
        """Tensor type hint for host context lengths with shape (batch_size,)."""
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field
    @property
    def host_max_attention_window_sizes(self) -> TensorTypeHint:
        """Tensor type hint for host max attention window sizes with shape (num_attn_layers_per_pipeline,)."""
        return TensorTypeHint(
            shape=(
                DynamicDimension(
                    name="host_max_attention_window_sizes",
                    min=self.num_attn_layers_per_pipeline,
                    opt=self.num_attn_layers_per_pipeline,
                    max=self.num_attn_layers_per_pipeline,
                ),
            ),
            dtype=torch.int32,
        )

    @computed_field
    @property
    def host_sink_token_length(self) -> TensorTypeHint:
        """Tensor type hint for host sink token length with shape (1,)."""
        return TensorTypeHint(shape=(self._one,), dtype=torch.int32)

    @computed_field
    @property
    def cache_indirection(self) -> TensorTypeHint:
        """Tensor type hint for cache indirection with shape (batch_size, beam_width, attention_window_size)."""
        return TensorTypeHint(
            shape=(self.batch_size, self.beam_width, self.max_len),
            dtype=torch.int32,
        )

    @computed_field
    @property
    def host_kv_cache_pool_mapping(self) -> TensorTypeHint:
        """Tensor type hint for host KV cache pool mapping with shape (num_attn_layers_per_pipeline, 2)."""
        return TensorTypeHint(
            shape=(
                DynamicDimension(
                    name="host_max_attention_window_sizes",
                    min=self.num_attn_layers_per_pipeline,
                    opt=self.num_attn_layers_per_pipeline,
                    max=self.num_attn_layers_per_pipeline,
                ),
                2,
            ),
            dtype=torch.int32,
        )

    @computed_field
    @property
    def host_context_progress(self) -> TensorTypeHint:
        """Tensor type hint for host context progress with shape (1,)."""
        return TensorTypeHint(shape=(self._one,), dtype=torch.int64)

    @computed_field
    @property
    def all_reduce_workspace(self) -> TensorTypeHint | None:
        """Tensor type hint for all reduce workspace with shape (workspace_size,) or None if tp_size is 1."""
        if self.mapping.tp_size == 1:
            return None
        pointers_per_rank = 7
        pointers_of_counter = 2
        workspace_size = pointers_per_rank * self.mapping.tp_size + pointers_of_counter
        return TensorTypeHint(
            shape=(
                DynamicDimension(
                    name="workspace_size",
                    min=workspace_size,
                    opt=workspace_size,
                    max=workspace_size,
                ),
            ),
            dtype=torch.int64,
        )

    @model_serializer(mode="wrap")
    def serialize_model(
        self,
        original_serializer: Callable[[Self], dict[str, TensorTypeHint | None]],
    ) -> dict[str, TensorTypeHint | None]:
        """Serialize model including LoRA input hints.

        Args:
            original_serializer (Callable[[Self], dict[str, TensorTypeHint | None]]): Original serializer function

        Returns:
            dict[str, TensorTypeHint | None]: Serialized model with LoRA and MRoPE hints
        """
        data = original_serializer(self)
        data.update(self.lora_input_hints)
        data.update(self.mrope_input_hints)
        return data
