from collections.abc import Generator

from loguru import logger
from torch.fx import GraphModule

from ...literals import LoraPluginInputPrefix
from ..subgraphs import Linear
from .infra import GraphOptimizationPass, PassResult


class BindUnmatchedLoraProtos(GraphOptimizationPass):
    """Bind unmatched LoRA prototypes to the correct layer."""

    def call(self, graph_module: GraphModule) -> PassResult:
        layer_index: int | None = None
        lora_prefix_generator: Generator[LoraPluginInputPrefix, None, None] | None = None
        for node in graph_module.graph.nodes:
            if not (linear := Linear.configure_from(node)):
                continue

            if layer_index is None or linear.layer_index != layer_index:
                lora_prefix_generator = generate_remaining_lora_prefixes()
                layer_index = linear.layer_index

            if lora_prefix_generator is None or linear.free_lora_proto is None:
                continue

            try:
                prefix = next(lora_prefix_generator)
            except StopIteration:
                logger.warning(f"The maximum number of Lora plugins per layer is reached at layer {layer_index}.")
                continue
            logger.warning(
                f"Failed to identify the transformer decoder layout of {linear.mm}. "
                f"Will use the prefix '{prefix}' for the LoRA plugin."
            )
            linear.bind_free_lora_proto(with_prefix=prefix)
        return PassResult(graph_module=graph_module, modified=False)


def generate_remaining_lora_prefixes() -> Generator[LoraPluginInputPrefix, None, None]:
    """Generate remaining LoRA prefixes that could be used for unmatched linear layers.

    This function yields prefixes that has not been used by `FuseQKVProjections` or `FuseGatedMLPProjections`.

    Yields:
        LoraPluginInputPrefix: LoRA plugin input prefixes
    """
    yield from (
        "cross_attn_qkv",
        "cross_attn_q",
        "cross_attn_k",
        "cross_attn_v",
        "cross_attn_dense",
        "moe_h_to_4h",
        "moe_4h_to_h",
        "moe_gate",
        "moe_router",
        "mlp_router",
    )
