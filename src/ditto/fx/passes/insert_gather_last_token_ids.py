import torch
from loguru import logger
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from ...constants import INPUT_IDS_UNSQUEEZE_DIM
from ..nodes import SymSizeInt
from ..subgraphs import Linear
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .graph_pass import GraphOptimizationPass


class InsertGatherLastTokenIds(GraphOptimizationPass):
    """Insert gather op for lm_head output using last_token_ids (required for trtllm)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        placeholders = {n.name: n for n in graph.find_nodes(op="placeholder")}
        input_ids_major_axis = 1 - INPUT_IDS_UNSQUEEZE_DIM
        if not (
            (last_token_ids := placeholders.get("last_token_ids"))
            and not any(user.target is torch.ops.aten.index_select.default for user in last_token_ids.users)
            and (last_token_ids_meta := get_tensor_metadata(last_token_ids))
            and isinstance(batch_size := last_token_ids_meta.shape[0], torch.SymInt)
            and (num_tokens_node := find_or_create_placeholder_sym_size(graph, "input_ids"))
            and (batch_size_node := find_or_create_placeholder_sym_size(graph, "last_token_ids"))
            and (lm_head := find_lm_head(graph))
            and (input_tensor_meta := get_tensor_metadata(lm_head.input_node))
            and isinstance(input_ids_size := input_tensor_meta.shape[input_ids_major_axis], torch.SymInt)
        ):
            return PassResult(graph_module, False)

        with graph.inserting_before(lm_head.input_reshape.node):
            sub = graph.call_function(torch.ops.aten.sub.Scalar, (last_token_ids, 1))
            populate_tensor_metadata(sub, last_token_ids_meta)
            index_select = graph.call_function(
                torch.ops.aten.index_select.default,
                (lm_head.input_node, input_ids_major_axis, sub),
            )
            populate_tensor_metadata(
                index_select,
                input_tensor_meta,
                shape=(
                    *input_tensor_meta.shape[:input_ids_major_axis],
                    batch_size,
                    *input_tensor_meta.shape[input_ids_major_axis + 1 :],
                ),
            )
        lm_head.input_reshape.node.replace_input_with(lm_head.input_node, index_select)
        node_indices = {node: i for i, node in enumerate(graph.nodes)}
        num_tokens_node.replace_all_uses_with(
            batch_size_node, delete_user_cb=lambda node: node_indices.get(node, -1) > node_indices[index_select]
        )
        replace_size(index_select, input_ids_size, batch_size)
        lm_head.output_reshape.node.replace_input_with(num_tokens_node, batch_size_node)
        return PassResult(graph_module, True)


def replace_size(node: Node, target: torch.SymInt, replacement: torch.SymInt) -> None:
    if tensor_meta := get_tensor_metadata(node):
        populate_tensor_metadata(
            node,
            tensor_meta,
            shape=torch.Size(replacement if s == target else s for s in tensor_meta.shape),  # type: ignore
        )
    for user in node.users:
        replace_size(user, target, replacement)


def find_lm_head(graph: Graph) -> Linear | None:
    nodes = list(graph.nodes)
    for node in nodes[::-1]:
        if subgraph := Linear.configure_from(node):
            return subgraph
    return None


def find_or_create_placeholder_sym_size(graph: Graph, name: str, dim: int = 0) -> Node | None:
    if name not in (placeholders := {node.name: node for node in graph.find_nodes(op="placeholder")}):
        logger.warning(f"No such placholder: {name}")
        return None
    placeholder = placeholders[name]
    for user in placeholder.users:
        if (sym_size_int := SymSizeInt.specialize_from(user)) and sym_size_int.dim == dim:
            return user
    last_placeholder = [*placeholders.values()][-1]
    with graph.inserting_after(last_placeholder):
        node = graph.call_function(torch.ops.aten.sym_size.int, (placeholder, dim))
        if (metadata := get_tensor_metadata(placeholder)) and isinstance(s := metadata.shape[dim], torch.SymInt):
            node.meta["val"] = s
        return node
