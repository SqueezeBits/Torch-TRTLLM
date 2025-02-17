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

from pydantic import Field

from ...types import StrictlyTyped
from .builder import TensorRTBuilderConfig
from .flags import TensorRTNetworkCreationFlags


class TensorRTConfig(StrictlyTyped):
    """Configuration for TensorRT.

    Attributes:
        network_creation_flags (TensorRTNetworkCreationFlags): The network creation flags.
        builder_config (TensorRTBuilderConfig): The builder configuration.
    """

    network_creation_flags: TensorRTNetworkCreationFlags = Field(default_factory=TensorRTNetworkCreationFlags)
    builder_config: TensorRTBuilderConfig = Field(default_factory=TensorRTBuilderConfig)
