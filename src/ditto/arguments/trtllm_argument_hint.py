import torch
from pydantic import Field, TypeAdapter, computed_field
from typing_extensions import Self

from ..configs import TRTLLMOptimizationProfileConfig
from ..constants import INPUT_IDS_UNSQUEEZE_DIM
from ..types import StrictlyTyped
from .dynamic_dim import DynamicDimension, DynamicDimensionType
from .tensor_type_hint import TensorTypeHint


class TRTLLMArgumentHint(StrictlyTyped):
    batch_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_tokens: DynamicDimensionType = Field(frozen=True, exclude=True)
    kv_cache_block_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    beam_width: DynamicDimensionType | int = Field(frozen=True, exclude=True)
    attention_window_size: DynamicDimensionType = Field(frozen=True, exclude=True)
    num_attn_layers: int | None = Field(default=None, exclude=True, ge=0)

    @classmethod
    def configure(cls, profile_config: TRTLLMOptimizationProfileConfig) -> Self:
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
        )

    def as_dict(self) -> dict[str, TensorTypeHint]:
        return TypeAdapter(dict[str, TensorTypeHint]).validate_python(self.model_dump())

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
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kv_cache_block_offsets(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1, self.batch_size, 2, self.kv_cache_block_size), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_kv_cache_block_offsets(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1, self.batch_size, 2, self.kv_cache_block_size), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_kv_cache_pool_pointers(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(1, 2), dtype=torch.int64)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sequence_length(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_request_types(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_past_key_value_lengths(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def context_lengths(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_runtime_perf_knobs(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(16,), dtype=torch.int64)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def host_context_lengths(self) -> TensorTypeHint:
        return TensorTypeHint(shape=(self.batch_size,), dtype=torch.int32)

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
            shape=(self.batch_size, self.beam_width, self.attention_window_size),
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
