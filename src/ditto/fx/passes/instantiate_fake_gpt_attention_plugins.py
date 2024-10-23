import operator
from functools import reduce

import torch
from torch.fx import GraphModule, Node
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from ...fake_gpt_attention_plugin import FakeGPTAttentionPlugin, ROPEConfig
from ..utils import find_or_create_placeholder_sym_size, get_tensor_metadata, populate_metadata


def instantiate_fake_gpt_attention_plugins(graph_module: GraphModule) -> GraphModule:
    graph = graph_module.graph
    batch_size_node: Node | None = find_or_create_placeholder_sym_size(graph, "input_ids")

    layer_idx = -1
    for node in graph.nodes:
        if not (
            node.target is FakeGPTAttentionPlugin
            and (input_reshape := node.all_input_nodes[0]).target is torch.ops.aten.reshape.default
            and (output_reshape := [*node.users.keys()][0]).target is torch.ops.aten.reshape.default
            and len(input_reshape.all_input_nodes) == 2
            and (sym_size_int := input_reshape.all_input_nodes[1]).target is torch.ops.aten.sym_size.int
            and (qkv_cat := input_reshape.all_input_nodes[0]).target is torch.ops.aten.cat.default
            and len(qkv_cat.all_input_nodes) == 3
            and isinstance(qkv_cat_dim := qkv_cat.args[1], int)
            and (query := get_tensor_metadata(qkv_cat.all_input_nodes[0])) is not None
            and (key := get_tensor_metadata(qkv_cat.all_input_nodes[1])) is not None
            and (value := get_tensor_metadata(qkv_cat.all_input_nodes[2])) is not None
        ):
            continue

        layer_idx += 1
        if batch_size_node:
            sym_size_int.replace_all_uses_with(batch_size_node)

        flat_input_shape = torch.Size((query.shape[0], reduce(operator.mul, query.shape[1:])))
        qkv_shape = torch.Size(
            sum(x.shape[i] for x in (query, key, value)) if i == qkv_cat_dim else query.shape[i]
            for i in range(len(query.shape))
        )
        flat_qkv_shape = torch.Size((qkv_shape[0], reduce(operator.mul, qkv_shape[1:])))

        num_heads = query.shape[-3]
        hidden_size_per_head = query.shape[-1]
        num_k_heads = key.shape[-3]
        k_hidden_size_per_head = key.shape[-1]
        num_v_heads = value.shape[-3]
        v_hidden_size_per_head = value.shape[-1]
        if num_k_heads != num_v_heads:
            raise NotImplementedError(
                "The input key and value with different number of heads is not supported. "
                f"(key.shape: {key.shape}, value.shape: {value.shape})"
            )
        if not hidden_size_per_head == k_hidden_size_per_head == v_hidden_size_per_head:
            raise NotImplementedError(
                "The input query, key and value with different hidden sizes per head is not supported. "
                f"(query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape})"
            )
        extra_kwargs = (
            rope_config.model_dump() if isinstance(rope_config := node.meta.get("rope_config"), ROPEConfig) else {}
        )
        node.target = FakeGPTAttentionPlugin(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_k_heads,
            head_size=hidden_size_per_head,
            **extra_kwargs,
        )
        populate_metadata(node, query, flat_input_shape)
        with graph.inserting_before(node):
            new_input_reshape = graph.call_function(
                torch.ops.aten.reshape.default, (qkv_cat, (batch_size_node or sym_size_int, -1))
            )
            populate_metadata(new_input_reshape, query, flat_qkv_shape)
            input_reshape.replace_all_uses_with(new_input_reshape)

        with graph.inserting_after(node):
            new_output_reshape = graph.call_function(
                torch.ops.aten.reshape.default, (node, (batch_size_node or sym_size_int, *query.shape[1:]))
            )
            populate_metadata(new_output_reshape, query)
            output_reshape.replace_all_uses_with(new_output_reshape)

        populate_metadata(qkv_cat, query, qkv_shape)

    clean_up_graph_after_modifications(graph_module)
    return graph_module
