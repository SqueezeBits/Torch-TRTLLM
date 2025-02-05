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

from pydantic import model_validator
from typing_extensions import Self

from .model import TRTLLMModelConfig
from .optimization_profile import RuntimeTRTLLMOptimizationProfileConfig, TRTLLMOptimizationProfileConfig


class TRTLLMBuildConfig(RuntimeTRTLLMOptimizationProfileConfig, TRTLLMModelConfig):
    """Minimal subset of properties in `trtllm.BuildConfig` required at runtime."""

    @classmethod
    def merge(
        cls,
        profile_config: TRTLLMOptimizationProfileConfig,
        model_config: TRTLLMModelConfig,
    ) -> Self:
        """Create a new instance by merging the profile and model configurations.

        Args:
            profile_config (TRTLLMOptimizationProfileConfig): The profile configuration.
            model_config (TRTLLMModelConfig): The model configuration.

        Returns:
            Self: The merged configuration.
        """
        return cls.model_validate(
            {
                **profile_config.runtime().model_dump(),
                **model_config.model_dump(),
            }
        )

    @model_validator(mode="after")
    def check_context_mha_dependencies(self) -> Self:
        """Check conditions imposed by `context_mha`.

        The criterions and error messages are adopted from `tensorrt_llm._common.check_max_num_tokens`.
        While TRT-LLM tries to adjust the wrong values provided by the user, we will simply reject them.

        Returns:
            Self: The validated instance.
        """
        assert self.plugin_config.context_fmha or self.max_num_tokens >= self.max_input_len, (
            f"When {self.plugin_config.context_fmha=}, {self.max_num_tokens=}) "
            f"should be at least {self.max_input_len=}."
        )
        assert not self.plugin_config.context_fmha or self.max_num_tokens >= self.plugin_config.tokens_per_block, (
            f"When {self.plugin_config.context_fmha=}, {self.max_num_tokens=} "
            f"should be at least {self.plugin_config.tokens_per_block=}."
        )
        return self
