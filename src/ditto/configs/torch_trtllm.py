from pydantic import Field

from ditto.configs.trtllm.build import TRTLLMBuildConfig
from ditto.configs.trtllm.engine import TRTLLMEngineConfig
from ditto.configs.trtllm.pretrained import TRTLLMPretrainedConfig

from ..types import StrictlyTyped
from .tensorrt import TensorRTConfig
from .trtllm import TRTLLMModelConfig, TRTLLMOptimizationProfileConfig


class TorchTRTLLMConfig(StrictlyTyped):
    trt: TensorRTConfig = Field(default_factory=TensorRTConfig)
    profile: TRTLLMOptimizationProfileConfig = Field(default_factory=TRTLLMOptimizationProfileConfig)
    model: TRTLLMModelConfig = Field(default_factory=TRTLLMModelConfig)

    def get_engine_config(self, pretrained_config: TRTLLMPretrainedConfig) -> TRTLLMEngineConfig:
        return TRTLLMEngineConfig(
            pretrained_config=pretrained_config,
            build_config=TRTLLMBuildConfig.merge(self.profile, self.model),
        )
