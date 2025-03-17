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
from torch.fx import GraphModule

from ..fx.nodes.get_attr import GetAttr
from ..fx.targets.gpt_attention_plugin import GPTAttentionPlugin
from .io import open_debug_artifact, should_save_debug_artifacts


def save_mla_weights_for_debug(graph_module: GraphModule) -> None:
    """Save MLA weights for debugging purposes.

    The weights are saved as PyTorch tensors in separate files named "mla_weights_{idx}.pt"
    where idx is the index of the GPT attention node.

    Args:
        graph_module (GraphModule): The graph module containing GPT attention nodes
    """
    if not should_save_debug_artifacts():
        return
    gpt_attention_idx = -1
    for node in graph_module.graph.nodes:
        if isinstance(node.target, GPTAttentionPlugin) and node.target.is_mla_enabled:
            gpt_attention_idx += 1
            fused_q_proj = GetAttr.specialize_from(node.all_input_nodes[-3])
            q_b_proj = GetAttr.specialize_from(node.all_input_nodes[-2])
            kv_b_proj = GetAttr.specialize_from(node.all_input_nodes[-1])
            with open_debug_artifact(f"mla_weights_{gpt_attention_idx}.pt", "wb") as f:
                if f:
                    mla_weights = {
                        "fused_q_proj": fused_q_proj.tensor,
                        "q_b_proj": q_b_proj.tensor,
                        "kv_b_proj": kv_b_proj.tensor,
                    }
                    torch.save(
                        mla_weights,
                        f,
                    )
