from pydantic import computed_field
from tensorrt_llm.version import __version__ as trtllm_version

from ...types import StrictlyTyped
from .build import TRTLLMBuildConfig
from .pretrained import TRTLLMPretrainedConfig


class TRTLLMEngineConfig(StrictlyTyped):
    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> str:
        return trtllm_version

    pretrained_config: TRTLLMPretrainedConfig
    build_config: TRTLLMBuildConfig
