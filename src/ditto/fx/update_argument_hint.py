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
    """Update the argument hint from graph module.

    Args:
        argument_hint (TRTLLMArgumentHint): The argument hint to update
        graph_module (GraphModule): The graph module to update the argument hint
    """
    match_input_ids_dynamic_dims(argument_hint, graph_module)
    argument_hint.num_attn_layers = len(
        graph_module.graph.find_nodes(
            op="call_function",
            target=torch._C._nn.scaled_dot_product_attention,
        ),
    )


def match_input_ids_dynamic_dims(argument_hint: TRTLLMArgumentHint, graph_module: GraphModule) -> None:
    """Match the dynamic dimensions of the input IDs tensor.

    Args:
        argument_hint (TRTLLMArgumentHint): The argument hint to update
        graph_module (GraphModule): The graph module to update the argument hint
    """
    if num_tokens_sym_int := get_input_ids_dynamic_dim(graph_module):
        argument_hint.num_tokens_range.sym_int = num_tokens_sym_int
        with detailed_sym_node_str():
            logger.debug(f"Matched {repr(argument_hint.num_tokens_range)} to {num_tokens_sym_int}")
    else:
        logger.warning(f"Failed to match dynamic dimension of {INPUT_IDS}")


def get_input_ids_dynamic_dim(graph_module: GraphModule) -> torch.SymInt | None:
    """Get the dynamic dimension of the input IDs tensor.

    Args:
        graph_module (GraphModule): The graph module to get the dynamic dimension

    Returns:
        torch.SymInt | None: The dynamic dimension of the input IDs tensor
    """
    if (
        (placeholders := {p.name: p for p in graph_module.graph.find_nodes(op="placeholder")})
        and INPUT_IDS in placeholders
        and (meta := get_tensor_metadata(placeholders[INPUT_IDS]))
        and isinstance(sym_int := meta.shape[1 - INPUT_IDS_UNSQUEEZE_DIM], torch.SymInt)
    ):
        return sym_int
    return None
