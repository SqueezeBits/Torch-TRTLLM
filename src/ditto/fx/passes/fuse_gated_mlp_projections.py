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

from torch.fx import Node

from ...literals import LoraPluginInputPrefix
from ..subgraphs import GatedMLP, Linear, TrailingReformatPath
from .fuse_projections import FuseProjections


class FuseGatedMLPProjections(FuseProjections):
    """Fuse input projections of a gated MLP layer to a single Linear subgraph."""

    @property
    def fused_lora_prefix(self) -> LoraPluginInputPrefix | None:
        return "mlp_h_to_4h"

    def find_projections(self, node: Node) -> list[Linear]:
        if not (down_proj := Linear.configure_from(node)):
            return []
        root = TrailingReformatPath.configure_from(down_proj.input_node).top
        if not (gated_mlp := GatedMLP.configure_from(root)):
            return []
        down_proj.bind_free_lora_proto(with_prefix="mlp_4h_to_h")
        if gated_mlp.gate_proj.mm.node == gated_mlp.up_proj.mm.node:
            gated_mlp.up_proj.bind_free_lora_proto(with_prefix="mlp_h_to_4h")
            return []

        gated_mlp.up_proj.bind_free_lora_proto(with_prefix="mlp_h_to_4h")
        gated_mlp.gate_proj.bind_free_lora_proto(with_prefix="mlp_gate")
        return [gated_mlp.up_proj, gated_mlp.gate_proj]
