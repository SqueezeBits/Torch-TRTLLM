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
from .literals import LoraCheckpointLiteral


class TRTLLMLoraConfig(StrictlyTyped):
    lora_dir: list[str] = Field(default_factory=list)
    lora_ckpt_source: LoraCheckpointLiteral = "hf"
    max_lora_rank: int = 64
    lora_target_modules: list[str] = Field(default_factory=list)
    trtllm_modules_to_hf_modules: dict[str, str] = Field(default_factory=dict)
