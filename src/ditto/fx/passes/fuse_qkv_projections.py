from torch.fx import GraphModule, Node

from ...debug import save_for_debug
from ...literals import LoraPluginInputPrefix
from ..nodes import ScaledDotProductAttention
from ..subgraphs import Linear, TrailingReformatPath
from .fuse_projections import FuseProjections


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
            and (q_proj := Linear.find_nearest(sdpa.query))
            and (k_proj := Linear.find_nearest(sdpa.key))
            and (v_proj := Linear.find_nearest(sdpa.value))
        ):
            return []
        if q_proj.mm.node == k_proj.mm.node == v_proj.mm.node:
            q_proj.bind_free_lora_proto(with_prefix="attn_qkv")
        else:
            for prefix, proj in zip(("attn_q", "attn_k", "attn_v"), (q_proj, k_proj, v_proj)):
                proj.bind_free_lora_proto(with_prefix=prefix)
        attn_dense.bind_free_lora_proto(with_prefix="attn_dense")
        return [q_proj, k_proj, v_proj]
