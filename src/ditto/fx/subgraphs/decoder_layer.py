from torch.fx import Node
from typing_extensions import Self

from ..subgraphs.linear import Linear, find_nearest_linear_projection
from ..targets import GPTAttentionPlugin
from ..utils import find_closest_common_descendant, get_descendants_with_depth
from .subgraph import Subgraph


class DecoderLayer(Subgraph):
    """A pattern representing a decoder layer of GPT model.

    This pattern is used to extract the linear subgraphs based on the GPT attention plugin node.
    1) Find the GPTAttentionPlugin node
    2) Find the nearest linear subgraph to the GPTAttentionPlugin node
       by using find_nearest_linear_projection (QKV linear)
    3) Find the descendant linear subgraphs by using find_descendant_linears
       (Dense linear, MLP gate, MLP up-projection, MLP down-projection)

    Attributes:
        gpt_attn_plugin (Node): The GPT attention plugin node
        attn_qkv (Linear): The QKV linear layer
        attn_dense (Linear): The dense linear layer
        mlp_gate (Linear): The MLP gate linear layer
        mlp_up_proj (Linear): The MLP up-projection linear layer
        mlp_down_proj (Linear): The MLP down-projection linear layer
    """

    gpt_attn_plugin: Node
    attn_qkv: Linear
    attn_dense: Linear
    mlp_gate: Linear
    mlp_up_proj: Linear  # Note: mlp_gate and mlp_up_proj may be fused in the future
    mlp_down_proj: Linear

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        """Extract configuration from a GPT attention plugin node.

        Recursively analyzes the plugin node's ancestors and predecessors to find the associated
        linear/gemm operations for attention and feed forward network.

        Args:
            node: Node expected to be a GPTAttentionPlugin

        Returns:
            DecoderLayer if successfully extracted, None otherwise
        """
        if not (
            isinstance(node.target, GPTAttentionPlugin)
            and (attn_qkv := find_nearest_linear_projection(node)) is not None
            and (remaining_linears := find_descendant_linears(node)) is not None
        ):
            return None

        attn_dense, maybe_mlp_gate, maybe_mlp_up_proj, mlp_down_proj = remaining_linears
        mlp_common_descendant = find_closest_common_descendant(maybe_mlp_gate.input_node, maybe_mlp_up_proj.input_node)
        if mlp_common_descendant is None:
            return None

        expected_common_descendant = list(maybe_mlp_gate.output_node.users)[0]
        if expected_common_descendant == mlp_common_descendant:
            mlp_gate, mlp_up_proj = maybe_mlp_gate, maybe_mlp_up_proj
        else:
            mlp_gate, mlp_up_proj = maybe_mlp_up_proj, maybe_mlp_gate

        return cls(
            gpt_attn_plugin=node,
            attn_qkv=attn_qkv,
            attn_dense=attn_dense,
            mlp_gate=mlp_gate,
            mlp_up_proj=mlp_up_proj,
            mlp_down_proj=mlp_down_proj,
        )


def find_descendant_linears(node: Node) -> tuple[Linear, Linear, Linear, Linear] | None:
    """Find the descendant linear subgraphs from GPT attention plugin node.

    The layers found are expected to be a dense linear of attention layer and
    3 layers(gate, up-projection, down-projection) of feed forward network layer.

    Args:
        node: The node expected to be a GPTAttentionPlugin node for searching the descendant linear subgraphs

    Returns:
        The descendant linear subgraphs
    """
    if (
        not (
            descendant_linear_subgraphs := {
                subgraph: depth
                for node, depth in get_descendants_with_depth(node).items()
                if (subgraph := Linear.configure_from(node))
            }
        )
        and len(descendant_linear_subgraphs) < 4
    ):
        return None
    sorted_subgraphs = sorted(descendant_linear_subgraphs, key=lambda subgraph: descendant_linear_subgraphs[subgraph])
    return tuple(sorted_subgraphs[0:4])
