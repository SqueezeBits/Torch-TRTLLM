from loguru import logger
from torch.fx import Graph, GraphModule, Node

from ...constants import INPUT_IDS_UNSQUEEZE_DIM
from ..nodes import IndexSelect, Sub, SymSizeInt
from ..subgraphs import Linear
from ..utils import forget_all_descendant_fake_tensors
from .infra import GraphOptimizationPass, PassResult


class InsertGatherLastTokenIds(GraphOptimizationPass):
    """Insert gather op for lm_head output using last_token_ids (required for trtllm)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        placeholders = {n.name: n for n in graph.find_nodes(op="placeholder")}
        input_ids_major_axis = 1 - INPUT_IDS_UNSQUEEZE_DIM
        if not (
            (last_token_ids := placeholders.get("last_token_ids"))
            and (num_tokens_node := find_or_create_placeholder_sym_size(graph, "input_ids"))
            and (batch_size_node := find_or_create_placeholder_sym_size(graph, "last_token_ids"))
            and (lm_head := find_lm_head(graph))
        ):
            return PassResult(graph_module=graph_module, modified=False)

        with graph.inserting_before(lm_head.input_reshape.node):
            sub = Sub.create(graph, last_token_ids, 1)
            index_select = IndexSelect.create(graph, lm_head.input_node, input_ids_major_axis, sub).node
        lm_head.input_reshape.node.replace_input_with(lm_head.input_node, index_select)
        node_indices = {node: i for i, node in enumerate(graph.nodes)}
        num_tokens_node.replace_all_uses_with(
            batch_size_node, delete_user_cb=lambda node: node_indices.get(node, -1) > node_indices[index_select]
        )
        forget_all_descendant_fake_tensors(index_select)
        lm_head.output.replace_input_with(num_tokens_node, batch_size_node)
        return PassResult(graph_module=graph_module, modified=True, require_fake_tensor_prop=True)


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
        return SymSizeInt.create(graph, placeholder, dim).node
