from pydantic import model_validator
from typing_extensions import Self

from .model import TRTLLMModelConfig
from .optimization_profile import TRTLLMOptimizationProfileConfig


class TRTLLMBuildConfig(TRTLLMOptimizationProfileConfig, TRTLLMModelConfig):
    """Minimal subset of properties in `trtllm.BuildConfig` required at runtime."""

    @classmethod
    def merge(
        cls,
        profile_config: TRTLLMOptimizationProfileConfig,
        model_config: TRTLLMModelConfig,
    ) -> Self:
        return cls.model_validate({**profile_config.model_dump(), **model_config.model_dump()})

    @model_validator(mode="after")
    def check_context_mha_dependencies(self) -> Self:
        """Check conditions imposed by `context_mha`.

        The criterions and error messages are adopted from `tensorrt_llm._common.check_max_num_tokens`.
        While TRT-LLM tries to adjust the wrong values provided by the user, we will simply reject them.
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
