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

from ...literals import KVCacheTypeLiteral
from ...types import StrictlyTyped
from .lora import TRTLLMLoraConfig
from .plugin import TRTLLMPluginConfig


class TRTLLMModelConfig(StrictlyTyped):
    """Model configuration for TRT-LLM.

    Attributes:
        max_prompt_embedding_table_size: Maximum size of the prompt embedding table.
        kv_cache_type: Type of KV cache to use ('PAGED', 'CONTIGUOUS' or 'DISABLED').
        gather_context_logits: Whether to gather context logits.
        gather_generation_logits: Whether to gather generation logits.
        lora_config: Configuration for LoRA adapters.
        plugin_config: Configuration for TRT-LLM plugins.
    """

    max_prompt_embedding_table_size: int = 0
    kv_cache_type: KVCacheTypeLiteral = "PAGED"
    gather_context_logits: bool = False
    gather_generation_logits: bool = False
    lora_config: TRTLLMLoraConfig = Field(default_factory=TRTLLMLoraConfig)
    plugin_config: TRTLLMPluginConfig = Field(default_factory=TRTLLMPluginConfig)
