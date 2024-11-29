from pydantic import Field

from ...types import StrictlyTyped
from .literals import KVCacheTypeLiteral
from .lora import TRTLLMLoraConfig
from .plugin import TRTLLMPluginConfig


class TRTLLMModelConfig(StrictlyTyped):
    max_prompt_embedding_table_size: int = 0
    kv_cache_type: KVCacheTypeLiteral = "PAGED"
    gather_context_logits: bool = False
    gather_generation_logits: bool = False
    lora_config: TRTLLMLoraConfig = Field(default_factory=TRTLLMLoraConfig)
    plugin_config: TRTLLMPluginConfig = Field(default_factory=TRTLLMPluginConfig)
