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

import torch
from pydantic import Field, TypeAdapter, computed_field
from typing_extensions import Self

from ..configs import TRTLLMOptimizationProfileConfig
from ..constants import INPUT_IDS_UNSQUEEZE_DIM
from ..types import StrictlyTyped
from .dynamic_dim import DynamicDimension, DynamicDimensionType
from .tensor_type_hint import TensorTypeHint


# pylint: disable=too-many-public-methods
class TRTLLMArgumentHint(StrictlyTyped):
    """Argument hint for TensorRT-LLM.

    This class is used to generate argument hints for TensorRT-LLM.
    """

    batch_size_range: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_tokens: DynamicDimensionType = Field(frozen=True, exclude=True)
    kv_cache_block_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    beam_width: DynamicDimensionType | int = Field(frozen=True, exclude=True)
    attention_window_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_attn_layers: int | None = Field(default=None, exclude=True, ge=0)
    tp_size: int = Field(default=1, exclude=True, gt=0)

    @classmethod
    def configure(
        cls,
        profile_config: TRTLLMOptimizationProfileConfig,
        *,
        tp_size: int = 1,
    ) -> Self:
        """Configure the argument hint.

        Args:
            profile_config (TRTLLMOptimizationProfileConfig): The optimization profile configuration
            tp_size (int): The Tensor Parallelism size

        Returns:
            Self: The configured argument hint
        """
        batch_size_range = DynamicDimension(
            name="batch_size_range",
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
            batch_size_range=batch_size_range,
            num_tokens=num_tokens,
            kv_cache_block_size=kv_cache_block_size,
            beam_width=beam_width,
            attention_window_size=attention_window_size,
            tp_size=tp_size,
        )

    def as_dict(self) -> dict[str, TensorTypeHint | None]:
        return TypeAdapter(dict[str, TensorTypeHint | None]).validate_python(self.model_dump())

    @property
    def batched_input_ids(self) -> TensorTypeHint:
        if INPUT_IDS_UNSQUEEZE_DIM == 0:
            return TensorTypeHint(shape=(1, self.num_tokens), dtype=torch.int32)
        return TensorTypeHint(shape=(self.num_tokens, 1), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def input_ids(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.num_tokens,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def position_ids(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.num_tokens,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def last_token_ids(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size_range,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kv_cache_block_offsets(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1, self.batch_size_range, 2, self.kv_cache_block_size), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_kv_cache_block_offsets(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1, self.batch_size_range, 2, self.kv_cache_block_size), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_kv_cache_pool_pointers(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1, 2), dtype=torch.int64)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sequence_length(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size_range,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_request_types(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size_range,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_past_key_value_lengths(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size_range,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def context_lengths(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size_range,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_runtime_perf_knobs(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(16,), dtype=torch.int64)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_context_lengths(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size_range,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_max_attention_window_sizes(self) -> TensorTypeHint:
        assert self.num_attn_layers is not None, "num_attn_layers needs to be set for host_max_attention_window_sizes"
        return TensorTypeHint(shape=(self.num_attn_layers,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_sink_token_length(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_indirection(self) -> TensorTypeHint:
        return TensorTypeHint(
            shape=(self.batch_size_range, self.beam_width, self.attention_window_size),
            dtype=torch.int32,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_kv_cache_pool_mapping(self) -> TensorTypeHint:
        assert self.num_attn_layers is not None, "num_attn_layers needs to be set for host_kv_cache_pool_mapping"
        return TensorTypeHint(shape=(self.num_attn_layers,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_context_progress(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1,), dtype=torch.int64)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_reduce_workspace(self) -> TensorTypeHint | None:
        if self.tp_size == 1:
            return None
        pointers_per_rank = 7
        pointers_of_counter = 2
        workspace_size = pointers_per_rank * self.tp_size + pointers_of_counter
        return TensorTypeHint(shape=(workspace_size,), dtype=torch.int64)
