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

from loguru import logger
from torch.fx import Graph, GraphModule, Node

from ...constants import INPUT_IDS_UNSQUEEZE_DIM
from ..nodes import IndexSelect, SubScalar, SymSizeInt
from ..subgraphs.linear import Linear
from ..utils import forget_all_descendant_fake_tensors
from .infra import GraphOptimizationPass, PassResult


class InsertGatherLastTokenIds(GraphOptimizationPass):
    """Insert gather op for lm_head output using last_token_ids (required for trtllm)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        placeholders = {n.name: n for n in graph.find_nodes(op="placeholder")}
        if not (
            (last_token_ids := placeholders.get("last_token_ids"))
            and (num_tokens_node := find_or_create_placeholder_sym_size(graph, "input_ids"))
            and (batch_size_node := find_or_create_placeholder_sym_size(graph, "last_token_ids"))
            and (lm_head := Linear.find_last(graph))
        ):
            return PassResult(graph_module=graph_module, modified=False)

        with graph.inserting_before(lm_head.output_node):
            sub = SubScalar.create(graph, last_token_ids, 1)
            index_select = IndexSelect.create(graph, lm_head.input_node, INPUT_IDS_UNSQUEEZE_DIM, sub).node
        lm_head.mm.node.replace_input_with(lm_head.mm.this, index_select)
        num_tokens_node.replace_all_uses_with(batch_size_node, delete_user_cb=lambda node: node > index_select)
        forget_all_descendant_fake_tensors(index_select)
        return PassResult(graph_module=graph_module, modified=True, require_fake_tensor_prop=True)


def find_or_create_placeholder_sym_size(graph: Graph, name: str, dim: int = 0) -> Node | None:
    """Find or create a symbolic size node for a placeholder tensor.

    Args:
        graph (Graph): The computation graph to search/modify
        name (str): Name of the placeholder tensor
        dim (int, optional): Dimension to get size for. Defaults to 0.

    Returns:
        Node | None: The found or created symbolic size node if successful, None if placeholder not found
    """
    if name not in (placeholders := {node.name: node for node in graph.find_nodes(op="placeholder")}):
        logger.warning(f"No such placholder: {name}")
        return None
    placeholder = placeholders[name]
    for user in placeholder.users:
        if (sym_size_int := SymSizeInt.specialize_from(user)) and sym_size_int.dim == dim:
            return user
    last_placeholder = [*placeholders.values()][-1]
    with graph.inserting_after(last_placeholder):
        return SymSizeInt.create(graph, placeholder, dim).node
