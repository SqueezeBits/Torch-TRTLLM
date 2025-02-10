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

from ..configs import TRTLLMOptimizationProfileConfig
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
        num_attn_layers (int | None): Number of attention layers. Defaults to None.
        tp_size (int): Tensor parallel size. Defaults to 1.
        gather_context_logits (bool): Whether to gather context logits. Defaults to False.
        lora_input_hints (dict[str, TensorTypeHint]): LoRA input tensor hints. Defaults to empty dict.
    """

    batch_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    max_len: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_tokens: DynamicDimensionType = Field(frozen=True, exclude=True)
    max_blocks_per_seq: DynamicDimensionType = Field(frozen=True, exclude=True)
    beam_width: DynamicDimensionType | int = Field(frozen=True, exclude=True)
    num_attn_layers: int | None = Field(default=None, exclude=True, ge=0)
    tp_size: int = Field(default=1, exclude=True, gt=0)
    gather_context_logits: bool = Field(default=False, exclude=True)
    lora_input_hints: dict[str, TensorTypeHint] = Field(default_factory=dict, exclude=True)
    _one: DynamicDimension = PrivateAttr(default=DynamicDimension(name="one", min=1, opt=1, max=1))
    _two: DynamicDimension = PrivateAttr(default=DynamicDimension(name="two", min=2, opt=2, max=2))

    @classmethod
    def configure(
        cls,
        profile_config: TRTLLMOptimizationProfileConfig,
        *,
        gather_context_logits: bool,
        tp_size: int = 1,
    ) -> Self:
        """Configure the argument hint.

        Args:
            profile_config (TRTLLMOptimizationProfileConfig): The optimization profile configuration
            gather_context_logits (bool): Whether to gather context logits
            tp_size (int): The Tensor Parallelism size
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
            tp_size=tp_size,
            gather_context_logits=gather_context_logits,
        )

    def as_dict(self) -> dict[str, TensorTypeHint | None]:
        """Convert argument hints to dictionary.

        Returns:
            dict[str, TensorTypeHint | None]: Dictionary of tensor hints
        """
        return TypeAdapter(dict[str, TensorTypeHint | None]).validate_python(self.model_dump())

    def create_dynamic_dim(self, name: str, ranges: list[int]) -> DynamicDimension:
        """Create a dynamic dimension.

        Args:
            name (str): The name of the dynamic dimension
            ranges (list[int]): The ranges of the dynamic dimension

        Returns:
            DynamicDimension: The created dynamic dimension
        """
        assert len(ranges) == 1 or len(ranges) == 3, "ranges must be a list of one or three integers"
        if len(ranges) == 1:
            dim_range = (ranges[0], ranges[0], ranges[0])
        else:
            dim_range = (ranges[0], ranges[1], ranges[2])
        return DynamicDimension(name=name, min=dim_range[0], opt=dim_range[2], max=dim_range[1])

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
    def last_token_ids(self) -> TensorTypeHint | None:
        """Tensor type hint for last token IDs with shape (num_tokens,), None if gather context logits is True."""
        if self.gather_context_logits:
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
        return TensorTypeHint(shape=(self.create_dynamic_dim("host_runtime_perf_knobs", [16]),), dtype=torch.int64)

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
        return TensorTypeHint(
            shape=(self.create_dynamic_dim("host_max_attention_window_sizes", [self.num_attn_layers]),),
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
        """Tensor type hint for host KV cache pool mapping with shape (num_attn_layers,)."""
        assert self.num_attn_layers is not None, "num_attn_layers needs to be set for host_kv_cache_pool_mapping"
        return TensorTypeHint(
            shape=(self.create_dynamic_dim("host_max_attention_window_sizes", [self.num_attn_layers]),),
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
