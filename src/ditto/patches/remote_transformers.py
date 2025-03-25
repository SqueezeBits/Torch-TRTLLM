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

import importlib.util

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ditto.patches.patch import custom_patch

PATCH_TARGETS = [
    "DeepseekForCausalLM",
    "DeepseekV2ForCausalLM",
]


def apply_remote_transformers_patches(model_class: type[PreTrainedModel]) -> None:
    model_name = model_class.__name__
    module_path = model_class.__module__
    if model_name not in PATCH_TARGETS:
        return
    global modeling_deepseek

    if model_name == "DeepseekV2ForCausalLM":
        modeling_deepseek = importlib.import_module(module_path)
        assert hasattr(modeling_deepseek, "DeepseekV2MoE")

        @custom_patch(
            name=f"{module_path}.DeepseekV2MoE.forward",
            reason="resolving torch.export failure observed in deepseek-v2 forward method.",
            required=True,
            env_var_to_disable="DISABLE_DEEPSEEK_V2_FORWARD_PATCH",
        )
        def patch_deepseek_v2_moe_forward() -> None:
            modeling_deepseek.DeepseekV2MoE.forward = patched_deepseek_moe_forward

        @custom_patch(
            name=f"{module_path}.DeepseekV2Attention.forward",
            reason=(
                "resolving dynamo decomposition failure observed in deepseek-v2 attention forward method."
                "This patch can be removed once this PR(https://github.com/pytorch/TensorRT/pull/3420) is merged."
            ),
            required=True,
            env_var_to_disable="DISABLE_DEEPSEEK_V2_ATTENTION_FORWARD_PATCH",
        )
        def patch_deepseek_v2_attention_forward() -> None:
            # NOTE: This is a temporary patch until the PR(https://github.com/pytorch/TensorRT/pull/3420) is merged.
            # Once the PR is merged and used in ditto, we should remove this patch.
            modeling_deepseek.DeepseekV2Attention.forward = patched_deepseek_v2_attention_forward

    if model_name == "DeepseekForCausalLM":
        modeling_deepseek = importlib.import_module(module_path)
        assert hasattr(modeling_deepseek, "DeepseekMoE")

        @custom_patch(
            name=f"{module_path}.DeepseekMoE.forward",
            reason="resolving torch.export failure observed in deepseek-v1 forward method.",
            required=True,
            env_var_to_disable="DISABLE_DEEPSEEK_V1_FORWARD_PATCH",
        )
        def patch_deepseek_v1() -> None:
            modeling_deepseek.DeepseekMoE.forward = patched_deepseek_moe_forward


# fmt: off
# ruff: noqa
def patched_deepseek_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    identity = hidden_states
    selected_experts, routing_weights, _ = self.gate(hidden_states)
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=len(self.experts)).permute(2, 1, 0)

    for expert_idx in range(len(self.experts)):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    y = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y


def patched_deepseek_v2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: torch.Tensor | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )

    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    kv_seq_len = value_states.shape[-2]

    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    q_pe, k_pe = modeling_deepseek.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    query_states = torch.cat([q_nope, q_pe], dim=-1)

    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    k_pe = k_pe.expand(bsz, self.num_heads, q_len, self.qk_rope_head_dim)
    key_states = torch.cat([k_nope, k_pe], dim=-1)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_weights = (
        torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    assert attention_mask is not None
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
