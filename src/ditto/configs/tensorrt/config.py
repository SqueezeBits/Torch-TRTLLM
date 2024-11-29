from pydantic import Field

from ...types import StrictlyTyped
from .builder import TensorRTBuilderConfig
from .flags import TensorRTNetworkCreationFlags


class TensorRTConfig(StrictlyTyped):
    network_creation_flags: TensorRTNetworkCreationFlags = Field(default_factory=TensorRTNetworkCreationFlags)
    builder_config: TensorRTBuilderConfig = Field(default_factory=TensorRTBuilderConfig)
