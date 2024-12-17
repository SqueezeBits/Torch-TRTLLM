import math
from typing import Any

from loguru import logger
from pydantic import Field, PrivateAttr, computed_field, model_validator
from transformers import PretrainedConfig
from typing_extensions import Self

from ...types import StrictlyTyped
from .plugin import TRTLLMPluginConfig

DEFAULT_MAX_POS_EMBEDDING: int = 4096


class RuntimeTRTLLMOptimizationProfileConfig(StrictlyTyped):
    """A subset of properties in `trtllm.BuildConfig` related to optimization profile required at runtime."""

    max_input_len: int = Field(default=1024, gt=1)
    max_seq_len: int = Field(default=DEFAULT_MAX_POS_EMBEDDING, gt=1)
    max_batch_size: int = Field(default=256, gt=1)
    max_beam_width: int = Field(default=1, gt=0)
    max_num_tokens: int = Field(default=8192, multiple_of=8, gt=1)
    opt_batch_size: int = Field(default=128, gt=0)
    _opt_num_tokens: int | None = PrivateAttr(default=None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def opt_num_tokens(self) -> int:
        if self._opt_num_tokens is None:
            opt_num_tokens = min(self.max_num_tokens, self.max_batch_size * self.max_beam_width)
            self._opt_num_tokens = 8 * max(int(round(opt_num_tokens / 8)), 1)
        return self._opt_num_tokens

    @opt_num_tokens.setter
    def opt_num_tokens(self, value: Any) -> None:
        assert isinstance(
            value, int | None
        ), f"`opt_num_tokens` must have type `int` or be `None` but assigned with {value}"
        if value is None:
            self._opt_num_tokens = None
            return

        assert value > 0, f"`opt_num_tokens` must be positive but assigned with {value}"
        if value % 8 != 0:
            rounded_value = 8 * max(int(round(value / 8)), 1)
            logger.warning(
                "torch.export will impose `opt_num_tokens` to be a multiple of 8. "
                f"The assigned value ({value}) will be rounded to {rounded_value}, the closest multiple of 8."
            )
            value = rounded_value
        if value != (optimal_value := self.max_batch_size * self.max_beam_width):
            logger.warning(
                f"The manually set opt_num_token ({value}) is not equal to the optimal value: "
                f"`max_batch_size x max_beam_width = {self.max_batch_size} x {self.max_beam_width} "
                f"= {optimal_value}`."
            )
        self._opt_num_tokens = value

    @model_validator(mode="after")
    def check_runtime_attribute_dependencies(self) -> Self:
        """Check dependencies of attributes.

        The criterions and error messages are adopted from `tensorrt_llm._common.check_max_num_tokens`.
        While TRT-LLM tries to adjust the wrong values provided by the user, we will simply reject them.
        """
        assert self.max_num_tokens <= self.max_seq_len * self.max_batch_size, (
            f"{self.max_num_tokens=} shouldn't be greater than "
            f"`max_seq_len x max_batch_size = {self.max_seq_len} x {self.max_batch_size} = "
            f"{self.max_seq_len * self.max_batch_size}`."
        )
        assert (
            self.opt_num_tokens <= self.max_num_tokens
        ), f"{self.max_num_tokens=} shouldn't be less than {self.opt_num_tokens=}."
        assert (
            self.opt_batch_size <= self.max_batch_size
        ), f"{self.opt_batch_size=} must be at most {self.max_batch_size=}."
        assert self._opt_num_tokens is None or self._opt_num_tokens <= self.max_num_tokens
        if self.max_num_tokens > (upper_bound := 16384):
            logger.warning(
                f"Specifying a {self.max_num_tokens=} larger than {upper_bound} is usually not recommended. You might "
                "miss performance gain and too large `max_num_tokens` could possibly exceed the TensorRT tensor volume."
            )
        return self


class TRTLLMOptimizationProfileConfig(RuntimeTRTLLMOptimizationProfileConfig):
    """A subset of properties in `trtllm.BuildConfig` related to optimization profile."""

    opt_seq_len: int = Field(default=2048, gt=0)
    opt_beam_width: int = Field(default=1, gt=0)
    opt_kv_cache_block_size: int = Field(default=32, gt=0)
    max_kv_cache_block_size: int = Field(default=64, gt=1)
    opt_attention_window_size: int = Field(default=DEFAULT_MAX_POS_EMBEDDING // 2, gt=0)
    max_attention_window_size: int = Field(default=DEFAULT_MAX_POS_EMBEDDING, gt=1)

    @classmethod
    def create_from(cls, hf_config: PretrainedConfig, plugin_config: TRTLLMPluginConfig) -> Self:
        max_position_embeddings = getattr(hf_config, "max_position_embeddings", DEFAULT_MAX_POS_EMBEDDING)
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            rotary_factor = rope_scaling.get("factor", 1.0)
            max_position_embeddings = math.ceil(max_position_embeddings * rotary_factor)
            logger.debug(f"max_seq_len is scaled to {max_position_embeddings} by rotary scaling {rotary_factor}.")
        max_kv_cache_block_size = math.ceil(max_position_embeddings / plugin_config.tokens_per_block)
        return cls(
            max_seq_len=max_position_embeddings,
            max_attention_window_size=max_position_embeddings,
            max_kv_cache_block_size=max_kv_cache_block_size,
            opt_kv_cache_block_size=max_kv_cache_block_size // 2,
        )

    def runtime(self) -> RuntimeTRTLLMOptimizationProfileConfig:
        runtime_config = RuntimeTRTLLMOptimizationProfileConfig(
            **{
                key: value
                for key, value in self.model_dump().items()
                # pylint: disable-next=unsupported-membership-test
                if key in RuntimeTRTLLMOptimizationProfileConfig.model_fields
            }
        )
        runtime_config.opt_num_tokens = self.opt_num_tokens
        return runtime_config

    @model_validator(mode="after")
    def check_attribute_dependencies(self) -> Self:
        """Check dependencies of attributes.

        The criterions and error messages are adopted from `tensorrt_llm._common.check_max_num_tokens`.
        While TRT-LLM tries to adjust the wrong values provided by the user, we will simply reject them.
        """
        assert self.opt_seq_len <= self.max_seq_len, f"{self.opt_seq_len=} must be at most {self.max_seq_len=}."
        assert self.opt_beam_width <= self.max_beam_width, f"{self.opt_seq_len=} must be at most {self.max_seq_len=}."
        assert (
            self.opt_kv_cache_block_size <= self.max_kv_cache_block_size
        ), f"{self.opt_kv_cache_block_size=} must be at most {self.max_kv_cache_block_size=}."
        assert (
            self.opt_attention_window_size <= self.max_attention_window_size
        ), f"{self.opt_attention_window_size=} must be at most {self.max_attention_window_size=}."
        assert (
            self.max_attention_window_size <= self.max_seq_len
        ), f"{self.max_attention_window_size=} must be at most {self.max_seq_len=}."
        return self
