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

from ...literals import PluginFlag
from ...types import StrictlyTyped


class TRTLLMPluginConfig(StrictlyTyped):
    """Plugin configuration for TRT-LLM.

    Attributes:
        gpt_attention_plugin (PluginFlag): Plugin flag for GPT attention. Default is "auto".
        gemm_plugin (PluginFlag): Plugin flag for GEMM. Default is "auto".
        lora_plugin (PluginFlag): Plugin flag for LoRA. Default is None.
        mamba_conv1d_plugin (PluginFlag): Plugin flag for Mamba Conv1D. Default is "auto".
        nccl_plugin (PluginFlag): Plugin flag for NCCL. Default is None.
        fp8_rowwise_gemm_plugin (PluginFlag): Plugin flag for FP8 rowwise GEMM. Default is None.
        weight_only_groupwise_quant_matmul_plugin (PluginFlag): Plugin flag for weight-only groupwise quant matmul.
            Default is None.
        weight_only_quant_matmul_plugin (PluginFlag): Plugin flag for weight-only quant GEMM. Default is None.
        rmsnorm_quantization_plugin (PluginFlag): Plugin flag for RMSNorm quantization. Default is None.
        quantize_per_token_plugin (bool): Whether to use per-token quantization. Default is False.
        quantize_tensor_plugin (bool): Whether to use tensor quantization. Default is False.
        context_fmha (bool): Whether to use context FMHA. Default is True.
        paged_kv_cache (bool): Whether to use paged KV cache. Default is True.
        remove_input_padding (bool): Whether to remove input padding. Default is True.
        tokens_per_block (int): Number of tokens per block. Default is 64.
        use_paged_context_fmha (bool): Whether to use paged context FMHA. Default is False.
        paged_state (bool): Whether to use paged state. Default is False.
    """

    gpt_attention_plugin: PluginFlag = "auto"
    gemm_plugin: PluginFlag = "auto"
    lora_plugin: PluginFlag = None
    moe_plugin: PluginFlag = None
    mamba_conv1d_plugin: PluginFlag = "auto"
    nccl_plugin: PluginFlag = None
    fp8_rowwise_gemm_plugin: PluginFlag = None
    weight_only_groupwise_quant_matmul_plugin: PluginFlag = None
    weight_only_quant_matmul_plugin: PluginFlag = None
    smooth_quant_plugins: bool = False
    smooth_quant_gemm_plugin: PluginFlag = None
    layernorm_quantization_plugin: PluginFlag = None
    rmsnorm_quantization_plugin: PluginFlag = None
    quantize_per_token_plugin: bool = False
    quantize_tensor_plugin: bool = False
    context_fmha: bool = True
    paged_kv_cache: bool = True
    remove_input_padding: bool = True
    tokens_per_block: int = 64
    use_paged_context_fmha: bool = False
    paged_state: bool = False

    @classmethod
    def create_from(
        cls,
        dtype: torch.dtype,
        world_size: int = 1,
        *,
        use_paged_context_fmha: bool = True,
    ) -> Self:
        """Create a plugin configuration from a given dtype and world size.

        Args:
            dtype: The dtype to create the plugin configuration for.
            world_size: The world size to create the plugin configuration for.
            use_paged_context_fmha: Whether to use paged context FMHA.

        Returns:
            The plugin configuration.
        """
        plugin_flag = get_plugin_flag(dtype)
        return cls(
            gpt_attention_plugin=plugin_flag,
            gemm_plugin=plugin_flag,
            nccl_plugin=plugin_flag if world_size > 1 else None,
            use_paged_context_fmha=use_paged_context_fmha,
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
