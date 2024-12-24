import torch
from loguru import logger
from torch.fx import GraphModule

from ..arguments import TRTLLMArgumentHint
from ..constants import INPUT_IDS, INPUT_IDS_UNSQUEEZE_DIM
from ..contexts import detailed_sym_node_str
from .utils import get_tensor_metadata


def update_argument_hint(
    argument_hint: TRTLLMArgumentHint,
    graph_module: GraphModule,
) -> None:
    match_input_ids_dynamic_dims(argument_hint, graph_module)
    argument_hint.num_attn_layers = len(
        graph_module.graph.find_nodes(
            op="call_function",
            target=torch._C._nn.scaled_dot_product_attention,
        ),
    )


def match_input_ids_dynamic_dims(argument_hint: TRTLLMArgumentHint, graph_module: GraphModule) -> None:
    if num_tokens_sym_int := get_input_ids_dynamic_dim(graph_module):
        argument_hint.num_tokens.sym_int = num_tokens_sym_int
        with detailed_sym_node_str():
            logger.debug(f"Matched {repr(argument_hint.num_tokens)} to {num_tokens_sym_int}")
    else:
        logger.warning(f"Failed to match dynamic dimension of {INPUT_IDS}")


def get_input_ids_dynamic_dim(graph_module: GraphModule) -> torch.SymInt | None:
    if (
        (placeholders := {p.name: p for p in graph_module.graph.find_nodes(op="placeholder")})
        and INPUT_IDS in placeholders
        and (meta := get_tensor_metadata(placeholders[INPUT_IDS]))
        and isinstance(sym_int := meta.shape[1 - INPUT_IDS_UNSQUEEZE_DIM], torch.SymInt)
    ):
        return sym_int
    return None
