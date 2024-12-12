import torch
from loguru import logger
from typing_extensions import Self

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

    @classmethod
    def create_from(cls, dtype: torch.dtype) -> Self:
        plugin_flag = get_plugin_flag(dtype)
        return cls(
            gpt_attention_plugin=plugin_flag,
            gemm_plugin=plugin_flag,
        )


def get_plugin_flag(dtype: torch.dtype) -> PluginFlag:
    mapping: dict[torch.dtype, PluginFlag] = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "fp8",
    }
    if dtype not in mapping:
        logger.warning(f"{dtype} is not supported for TRT-LLM plugins. Will use the plugin flag 'auto' instead.")
        return "auto"
    return mapping[dtype]
