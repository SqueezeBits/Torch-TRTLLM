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
        gated_mlp.gate_proj.bind_free_lora_proto(with_prefix="mlp_gate")
        gated_mlp.up_proj.bind_free_lora_proto(with_prefix="mlp_h_to_4h")
        down_proj.bind_free_lora_proto(with_prefix="mlp_4h_to_h")
        return [gated_mlp.up_proj, gated_mlp.gate_proj]
