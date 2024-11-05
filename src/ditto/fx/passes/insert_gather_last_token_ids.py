import logging

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from ...config import INPUT_IDS_UNSQUEEZE_DIM
from ..subgraphs import LinearSubgraph
from ..utils import find_or_create_placeholder_sym_size, get_tensor_metadata, populate_tensor_metadata
from .graph_pass import GraphOptimizationPass

logger = logging.getLogger(__name__)


class InsertGatherLastTokenIds(GraphOptimizationPass):
    """Insert gather op for lm_head output using last_token_ids (required for trtllm)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        placeholders = {n.name: n for n in graph.nodes if n.op == "placeholder"}
        input_ids_major_axis = 1 - INPUT_IDS_UNSQUEEZE_DIM
        if not (
            (last_token_ids := placeholders.get("last_token_ids"))
            and not any(user.target is torch.ops.aten.index_select.default for user in last_token_ids.users)
            and (last_token_ids_meta := get_tensor_metadata(last_token_ids))
            and isinstance(num_seq := last_token_ids_meta.shape[0], torch.SymInt)
            and (input_ids_size_node := find_or_create_placeholder_sym_size(graph, "input_ids"))
            and (num_seq_node := find_or_create_placeholder_sym_size(graph, "last_token_ids"))
            and (lm_head := find_lm_head(graph))
            and (input_tensor_meta := get_tensor_metadata(lm_head.input_tensor))
            and isinstance(input_ids_size := input_tensor_meta.shape[input_ids_major_axis], torch.SymInt)
        ):
            return PassResult(graph_module, False)

        with graph.inserting_before(lm_head.input_reshape):
            sub = graph.call_function(torch.ops.aten.sub.Scalar, (last_token_ids, 1))
            populate_tensor_metadata(sub, last_token_ids_meta)
            index_select = graph.call_function(
                torch.ops.aten.index_select.default,
                (lm_head.input_tensor, input_ids_major_axis, sub),
            )
            populate_tensor_metadata(
                index_select,
                input_tensor_meta,
                shape=(
                    *input_tensor_meta.shape[:input_ids_major_axis],
                    num_seq,
                    *input_tensor_meta.shape[input_ids_major_axis + 1 :],
                ),
            )
        lm_head.input_tensor.replace_all_uses_with(
            index_select,
            delete_user_cb=lambda node: node is not index_select,
        )
        node_indices = {node: i for i, node in enumerate(graph.nodes)}
        input_ids_size_node.replace_all_uses_with(
            num_seq_node, delete_user_cb=lambda node: node_indices.get(node, -1) > node_indices[index_select]
        )
        replace_size(index_select, input_ids_size, num_seq)
        return PassResult(graph_module, True)


def replace_size(node: Node, target: torch.SymInt, replacement: torch.SymInt) -> None:
    if tensor_meta := get_tensor_metadata(node):
        populate_tensor_metadata(
            node,
            tensor_meta,
            shape=torch.Size(replacement if s == target else s for s in tensor_meta.shape),  # type: ignore
        )
        if "val" in node.meta:
            _ = node.meta.pop("val")
    for user in node.users:
        replace_size(user, target, replacement)


def find_lm_head(graph: Graph) -> LinearSubgraph | None:
    nodes = list(graph.nodes)
    for node in nodes[::-1]:
        if subgraph := LinearSubgraph.configure_from(node):
            return subgraph
    return None
