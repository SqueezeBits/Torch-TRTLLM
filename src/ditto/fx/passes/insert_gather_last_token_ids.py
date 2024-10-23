import logging

import torch
from torch.fx import Graph, GraphModule, Node
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications
from typing_extensions import Self

from ...types import StrictlyTyped
from ..utils import find_or_create_placeholder_sym_size, get_tensor_metadata, populate_metadata

logger = logging.getLogger(__name__)


def insert_gather_last_token_ids(graph_module: GraphModule) -> GraphModule:
    graph = graph_module.graph
    placeholders = {n.name: n for n in graph.nodes if n.op == "placeholder"}
    if not (
        (last_token_ids := placeholders.get("last_token_ids"))
        and (last_token_ids_meta := get_tensor_metadata(last_token_ids))
        and isinstance(num_seq := last_token_ids_meta.shape[0], torch.SymInt)
        and (batch_size := find_or_create_placeholder_sym_size(graph, "input_ids"))
        and (num_seq_node := find_or_create_placeholder_sym_size(graph, "last_token_ids"))
        and (lm_head := find_lm_head(graph))
        and (input_tensor_meta := get_tensor_metadata(lm_head.input_tensor))
    ):
        return graph_module

    with graph.inserting_before(lm_head.input_reshape):
        index_select = graph.call_function(
            torch.ops.aten.index_select.default,
            (lm_head.input_tensor, 0, last_token_ids),
        )
        shape = torch.Size((num_seq, *input_tensor_meta.shape[1:]))
        populate_metadata(
            index_select,
            input_tensor_meta,
            shape,
        )
    lm_head.input_tensor.replace_all_uses_with(
        index_select,
        delete_user_cb=lambda node: node is not index_select,
    )
    node_indices = {node: i for i, node in enumerate(graph.nodes)}
    batch_size.replace_all_uses_with(
        num_seq_node, delete_user_cb=lambda node: node_indices.get(node, -1) > node_indices[index_select]
    )
    fix_batch_size(index_select, num_seq)
    clean_up_graph_after_modifications(graph_module)
    return graph_module


class LinearSubgraph(StrictlyTyped):
    mm: Node
    weight: Node
    input_reshape: Node
    output_reshape: Node

    @classmethod
    def configure_from(cls, mm_default: Node) -> Self | None:
        if (
            mm_default.op == "call_function"
            and mm_default.target is torch.ops.aten.mm.default
            and len(mm_default.all_input_nodes) == 2
            and (input_reshape := mm_default.all_input_nodes[0]).op == "call_function"
            and input_reshape.target is torch.ops.aten.reshape.default
            and (weight := mm_default.all_input_nodes[1]).op == "get_attr"
            and len(mm_default.users) == 1
            and (output_reshape := [*mm_default.users][0]).op == "call_function"
            and output_reshape.target is torch.ops.aten.reshape.default
        ):
            return cls(
                mm=mm_default,
                weight=weight,
                input_reshape=input_reshape,
                output_reshape=output_reshape,
            )
        return None

    @property
    def input_tensor(self) -> Node:
        return self.input_reshape.all_input_nodes[0]

    @property
    def users(self) -> dict[Node, None]:
        return self.output_reshape.users


def fix_batch_size(node: Node, new_batch_size: torch.SymInt) -> None:
    if tensor_meta := get_tensor_metadata(node):
        populate_metadata(node, tensor_meta, torch.Size([new_batch_size, *tensor_meta.shape[1:]]))  # type: ignore
        if "val" in node.meta:
            _ = node.meta.pop("val")
    for user in node.users:
        fix_batch_size(user, new_batch_size)


def find_lm_head(graph: Graph) -> LinearSubgraph | None:
    nodes = list(graph.nodes)
    for node in nodes[::-1]:
        if subgraph := LinearSubgraph.configure_from(node):
            return subgraph
    return None
