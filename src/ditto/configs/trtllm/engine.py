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

from pydantic import computed_field
from tensorrt_llm.version import __version__ as trtllm_version

from ...types import StrictlyTyped
from .build import TRTLLMBuildConfig
from .pretrained import TRTLLMPretrainedConfig


class TRTLLMEngineConfig(StrictlyTyped):
    """The configuration for the TensorRT-LLM engine.

    Attributes:
        pretrained_config (TRTLLMPretrainedConfig): The pretrained configuration.
        build_config (TRTLLMBuildConfig): The build configuration.
    """

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> str:
        """Get the version of the TensorRT-LLM engine.

        Returns:
            str: The version of the TensorRT-LLM engine.
        """
        return trtllm_version

    pretrained_config: TRTLLMPretrainedConfig
    build_config: TRTLLMBuildConfig
