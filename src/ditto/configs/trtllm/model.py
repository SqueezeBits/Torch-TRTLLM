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
from tensorrt_llm.models import SpeculativeDecodingMode

from ...literals import DTypeLiteral, KVCacheTypeLiteral
from ...types import StrictlyTyped
from .lora import TRTLLMLoraConfig
from .plugin import TRTLLMPluginConfig


class TRTLLMModelConfig(StrictlyTyped):
    """A subset of properties in `trtllm.BuildConfig` related to model configuration required at runtime.

    Attributes:
        kv_cache_type (KVCacheTypeLiteral): Type of KV cache to use ('PAGED', 'CONTIGUOUS' or 'DISABLED').
        logits_dtype (DTypeLiteral): Data type for logits.
        gather_context_logits (bool): Whether to gather context logits.
        gather_generation_logits (bool): Whether to gather generation logits.
        lora_config (TRTLLMLoraConfig): Configuration for LoRA adapters.
        plugin_config (TRTLLMPluginConfig): Configuration for TRT-LLM plugins.
        use_mrope (bool): Whether to use MRoPE.
        speculative_decoding_mode (SpeculativeDecodingMode): Mode of speculative decoding.
        max_draft_len (int): Maximum lengths of draft tokens for speculative decoding target model.
    """

    kv_cache_type: KVCacheTypeLiteral = "PAGED"
    logits_dtype: DTypeLiteral = "float32"
    gather_context_logits: bool = False
    gather_generation_logits: bool = False
    lora_config: TRTLLMLoraConfig = Field(default_factory=TRTLLMLoraConfig)
    plugin_config: TRTLLMPluginConfig = Field(default_factory=TRTLLMPluginConfig)
    use_mrope: bool = False
    speculative_decoding_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE
    max_draft_len: int = 0
