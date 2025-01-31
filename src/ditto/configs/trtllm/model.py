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
