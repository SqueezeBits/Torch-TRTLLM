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

import torch
from loguru import logger
from typing_extensions import Self

from ...types import StrictlyTyped
from .literals import PluginFlag


class TRTLLMPluginConfig(StrictlyTyped):
    """Plugin configuration for TRT-LLM.

    Attributes:
        gpt_attention_plugin: Plugin flag for GPT attention.
        gemm_plugin: Plugin flag for GEMM.
        lora_plugin: Plugin flag for LoRA.
        mamba_conv1d_plugin: Plugin flag for Mamba Conv1D.
        nccl_plugin: Plugin flag for NCCL.
        context_fmha: Whether to use context FMHA.
        paged_kv_cache: Whether to use paged KV cache.
        remove_input_padding: Whether to remove input padding.
        tokens_per_block: Number of tokens per block.
        use_paged_context_fmha: Whether to use paged context FMHA.
        paged_state: Whether to use paged state.
    """

    gpt_attention_plugin: PluginFlag = "auto"
    gemm_plugin: PluginFlag = "auto"
    lora_plugin: PluginFlag = None
    mamba_conv1d_plugin: PluginFlag = "auto"
    nccl_plugin: PluginFlag = None
    context_fmha: bool = True
    paged_kv_cache: bool = True
    remove_input_padding: bool = True
    tokens_per_block: int = 64
    use_paged_context_fmha: bool = False
    paged_state: bool = False

    @classmethod
    def create_from(cls, dtype: torch.dtype, world_size: int = 1) -> Self:
        """Create a plugin configuration from a given dtype and world size.

        Args:
            dtype: The dtype to create the plugin configuration for.
            world_size: The world size to create the plugin configuration for.

        Returns:
            The plugin configuration.
        """
        plugin_flag = get_plugin_flag(dtype)
        return cls(
            gpt_attention_plugin=plugin_flag,
            gemm_plugin=plugin_flag,
            nccl_plugin=plugin_flag if world_size > 1 else None,
        )


def get_plugin_flag(dtype: torch.dtype) -> PluginFlag:
    """Get the plugin flag for a given dtype.

    Args:
        dtype: The dtype to get the plugin flag for.

    Returns:
        The plugin flag for the given dtype.
    """
    mapping: dict[torch.dtype, PluginFlag] = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "fp8",
    }
    if dtype not in mapping:
        logger.warning(f"{dtype} is not supported for TRT-LLM plugins. Will use the plugin flag 'auto' instead.")
        return "auto"
    return mapping[dtype]
