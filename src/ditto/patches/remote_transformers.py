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
from transformers import PreTrainedModel

from ditto.patches.patch import custom_patch

PATCH_TARGETS = [
    "DeepseekForCausalLM",
    "DeepseekV2ForCausalLM",
]


def apply_model_specific_patch(model: PreTrainedModel) -> None:
    model_name = model.__class__.__name__
    module_path = type(model).__module__
    if model_name not in PATCH_TARGETS:
        return

    if model_name == "DeepseekV2ForCausalLM":
        modeling_deepseek = importlib.import_module(module_path)
        assert hasattr(modeling_deepseek, "DeepseekV2MoE")

        @custom_patch(
            name=f"{module_path}.DeepseekV2MoE.forward",
            reason="resolving torch.export failure observed in deepseek-v2 forward method.",
            required=True,
            env_var_to_disable="DISABLE_DEEPSEEK_V2_FORWARD_PATCH",
        )
        def patch_deepseek_v2() -> None:
            modeling_deepseek.DeepseekV2MoE.forward = patched_deepseek_moe_forward

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
