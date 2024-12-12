from ...types import StrictlyTyped
from .literals import PluginFlag


class TRTLLMPluginConfig(StrictlyTyped):
    gpt_attention_plugin: PluginFlag = "auto"
    gemm_plugin: PluginFlag = "auto"
    lora_plugin: PluginFlag = None
    mamba_conv1d_plugin: PluginFlag = "auto"
    context_fmha: bool = True
    paged_kv_cache: bool = True
    remove_input_padding: bool = True
    tokens_per_block: int = 64
    use_paged_context_fmha: bool = False
    paged_state: bool = False
