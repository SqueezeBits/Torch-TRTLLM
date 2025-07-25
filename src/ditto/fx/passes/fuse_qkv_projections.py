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

from torch.fx import GraphModule, Node

from ...debug import save_for_debug
from ...literals import LoraPluginInputPrefix
from ..nodes import ScaledDotProductAttention
from ..subgraphs import Linear, RmsNormSubgraph, TrailingReformatPath
from ..utils import find_nearest
from .fuse_projections import FuseProjections
from .replace_sdpa_by_gpt_attention_plugin import MLA


class FuseQKVProjections(FuseProjections):
    """Fuse input projections of an attention layer to a single Linear subgraph."""

    @property
    def fused_lora_prefix(self) -> LoraPluginInputPrefix | None:
        return "attn_qkv"

    def preprocess(self, graph_module: GraphModule) -> None:
        super().preprocess(graph_module)
        save_for_debug("before_qkv_fusion", graph_module)

    def find_projections(self, node: Node) -> list[Linear]:
        if not (attn_dense := Linear.configure_from(node)):
            return []
        root = TrailingReformatPath.configure_from(attn_dense.input_node).top
        if not (
            (sdpa := ScaledDotProductAttention.specialize_from(root))
            and (q_proj := find_nearest(Linear, sdpa.query))
            and (k_proj := find_nearest(Linear, sdpa.key))
            and (v_proj := find_nearest(Linear, sdpa.value))
        ):
            return []
        if find_nearest(RmsNormSubgraph, sdpa.query, max_depth=4) or find_nearest(
            RmsNormSubgraph, sdpa.key, max_depth=10
        ):
            # Note: If there are q_norm and k_norm, we don't fuse the projections.
            return []
        attn_dense.bind_free_lora_proto(with_prefix="attn_dense")
        if q_proj.mm.node == k_proj.mm.node == v_proj.mm.node:
            q_proj.bind_free_lora_proto(with_prefix="attn_qkv")
            return []

        for prefix, proj in zip(("attn_q", "attn_k", "attn_v"), (q_proj, k_proj, v_proj)):
            proj.bind_free_lora_proto(with_prefix=prefix)  # type: ignore
        return [q_proj, k_proj, v_proj]


class FuseMLAQKVProjections(FuseProjections):
    """Fuse input projections of a MLA layer to a single Linear subgraph."""

    @property
    def fused_lora_prefix(self) -> LoraPluginInputPrefix | None:
        return "attn_qkv"

    def preprocess(self, graph_module: GraphModule) -> None:
        super().preprocess(graph_module)
        save_for_debug("before_mlaqkv_fusion", graph_module)

    def find_projections(self, node: Node) -> list[Linear]:
        if not (
            (sdpa := ScaledDotProductAttention.specialize_from(node))
            and (mla := MLA.extract_from(sdpa))
            and (q_a_proj := mla.q_a_proj)
        ):
            return []

        return [q_a_proj, mla.kv_a_proj]
