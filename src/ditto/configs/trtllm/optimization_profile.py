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

import math

from loguru import logger
from pydantic import Field, model_validator
from transformers import PretrainedConfig
from typing_extensions import Self

from ...constants import DEFAULT_MAX_POS_EMBEDDING
from ...types import StrictlyTyped
from .plugin import TRTLLMPluginConfig


class RuntimeTRTLLMOptimizationProfileConfig(StrictlyTyped):
    """A subset of properties in `trtllm.BuildConfig` related to optimization profile required at runtime."""

    max_input_len: int = Field(default=1024, gt=0)
    max_seq_len: int = Field(default=DEFAULT_MAX_POS_EMBEDDING, gt=0)
    opt_batch_size: int = Field(default=128, gt=0)
    max_batch_size: int = Field(default=256, gt=0)
    max_beam_width: int = Field(default=1, gt=0)
    max_num_tokens: int = Field(default=8192, gt=0)
    opt_num_tokens: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def check_runtime_attribute_dependencies(self) -> Self:
        """Check dependencies of attributes.

        The criterions and error messages are adopted from `tensorrt_llm._common.check_max_num_tokens`.
        While TRT-LLM tries to adjust the wrong values provided by the user, we will simply reject them.
        """
        assert (
            self.max_input_len <= self.max_seq_len
        ), f"{self.max_input_len=} shouldn't be greater than {self.max_seq_len=}."
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

    @classmethod
    def create_from(
        cls,
        hf_config: PretrainedConfig,
        plugin_config: TRTLLMPluginConfig,
        *,
        max_batch_size: int = 256,
        max_seq_len: int | None = None,
        max_input_len: int = 1024,
        max_num_tokens: int = 8192,
        opt_num_tokens: int | None = None,
        max_beam_width: int = 1,
    ) -> Self:
        """Configure the optimization profile from given configurations.

        Args:
            hf_config (PretrainedConfig): The Hugging Face configuration
            plugin_config (TRTLLMPluginConfig): The TensorRT-LLM plugin configuration
            max_batch_size (int): The maximum batch size
            max_seq_len (int | None): The maximum sequence length
            max_input_len (int): The maximum input length
            max_num_tokens (int): The maximum number of tokens
            opt_num_tokens (int | None): The optimized number of tokens
            max_beam_width (int): The maximum beam width

        Returns:
            TRTLLMOptimizationProfileConfig: The optimization profile configuration
        """

        def divide_by_2(value: int, *, offset: int = 0) -> int:
            if (ret := (value + offset) // 2) == 0:
                return 1
            return ret

        max_position_embeddings = getattr(hf_config, "max_position_embeddings", DEFAULT_MAX_POS_EMBEDDING)
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            rotary_type = rope_scaling.get("type", rope_scaling.get("rope_type", None))
            rotary_factor = rope_scaling.get("factor", 1.0) if rotary_type not in ("su", "longrope", "llama3") else 1.0
            max_position_embeddings = math.ceil(max_position_embeddings * rotary_factor)
        if max_seq_len is None:
            max_seq_len = max_position_embeddings
            logger.debug(f"max_seq_len is not specified, using deduced value {max_seq_len}")
        if max_input_len > max_seq_len:
            logger.debug(
                f"max_input_len ({max_input_len}) is larger than max_seq_len ({max_seq_len}), using max_seq_len"
            )
            max_input_len = max_seq_len
        if opt_num_tokens is None:
            opt_num_tokens = min(max_num_tokens, max_batch_size * max_beam_width)
            logger.debug(f"opt_num_tokens is not set, specifying to {opt_num_tokens}")
        max_kv_cache_block_size = math.ceil(max_seq_len / plugin_config.tokens_per_block)
        opt_kv_cache_block_size = math.ceil(divide_by_2(max_seq_len) / plugin_config.tokens_per_block)
        return cls(
            max_batch_size=max_batch_size,
            opt_batch_size=divide_by_2(max_batch_size, offset=1),
            max_seq_len=max_seq_len,
            opt_seq_len=divide_by_2(max_seq_len, offset=1),
            max_input_len=max_input_len,
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            max_beam_width=max_beam_width,
            opt_beam_width=divide_by_2(max_beam_width, offset=1),
            max_kv_cache_block_size=max_kv_cache_block_size,
            opt_kv_cache_block_size=opt_kv_cache_block_size,
        )

    def runtime(self) -> RuntimeTRTLLMOptimizationProfileConfig:
        """Create a runtime configuration from the current configuration.

        Returns:
            RuntimeTRTLLMOptimizationProfileConfig: The runtime configuration
        """
        runtime_config = RuntimeTRTLLMOptimizationProfileConfig(
            **{
                key: value
                for key, value in self.model_dump().items()
                # pylint: disable-next=unsupported-membership-test
                if key in RuntimeTRTLLMOptimizationProfileConfig.model_fields
            }
        )
        return runtime_config

    @model_validator(mode="after")
    def check_attribute_dependencies(self) -> Self:
        """Check dependencies of attributes.

        The criterions and error messages are adopted from `tensorrt_llm._common.check_max_num_tokens`.
        While TRT-LLM tries to adjust the wrong values provided by the user, we will simply reject them.
        """
        assert self.opt_seq_len <= self.max_seq_len, f"{self.opt_seq_len=} must be at most {self.max_seq_len=}."
        assert (
            self.opt_beam_width <= self.max_beam_width
        ), f"{self.opt_beam_width=} must be at most {self.max_beam_width=}."
        assert (
            self.opt_kv_cache_block_size <= self.max_kv_cache_block_size
        ), f"{self.opt_kv_cache_block_size=} must be at most {self.max_kv_cache_block_size=}."
        return self
