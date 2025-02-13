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

import tensorrt as trt
import torch
from torch._subclasses import FakeTensor
from torch.fx import Graph, GraphModule, Node

from ...arguments import TRTLLMArgumentHint
from ...types import DataType
from ..nodes import AddTensorTensor, BinaryElementwise, Placeholder, SymSizeInt, Unsqueeze
from ..targets import GPTAttentionPlugin, RecvPlugin, SendPlugin
from ..utils import find_closest_common_descendant, get_val
from .infra import (
    GraphOptimizationPass,
    PassResult,
    cleanup,
    propagate_metadata_from,
)


# pylint: disable=too-many-locals
class ParallelizePipeline(GraphOptimizationPass):
    """Parallelize the pipeline in the graph (Pipeline Parallelism).

    Attributes:
        argument_hint (TRTLLMArgumentHint): The argument hint of the model
    """

    argument_hint: TRTLLMArgumentHint

    def call(self, graph_module: GraphModule) -> PassResult:
        overall_modified = False
        graph = graph_module.graph
        mapping = self.argument_hint.mapping
        assert (
            num_attn_layers := get_num_gpt_attn_plugin_nodes(graph)
        ) % mapping.pp_size == 0, (
            f"Number of attention layers ({num_attn_layers}) must be divisible by pipeline stages ({mapping.pp_size})"
        )
        layers_to_be_parallelized = mapping.get_pp_layers(num_attn_layers)
        for node in graph.nodes:
            if not (
                isinstance(gpt_attn_plugin := node.target, GPTAttentionPlugin)
                and gpt_attn_plugin.layer_idx in layers_to_be_parallelized
            ):
                continue

            if (
                not mapping.is_first_pp_rank()
                and gpt_attn_plugin.layer_idx == layers_to_be_parallelized[0]
                and (pipeline_input_node := find_input_node_of_pipeline(node))
                and (input_tensor := get_val(pipeline_input_node, FakeTensor)) is not None
                and (input_ids_node := find_input_ids_node(graph))
            ):
                self.argument_hint.hidden_size = input_tensor.shape[-1]
                self.argument_hint.hidden_dtype = input_tensor.dtype

                with graph.inserting_after(input_ids_node):
                    hidden_states_input = Placeholder.create(
                        graph, "hidden_states_input", self.argument_hint.hidden_states_input
                    )
                input_ids_node.replace_all_uses_with(
                    hidden_states_input.node,
                    delete_user_cb=lambda user: SymSizeInt.specialize_from(user) is None,
                )

                recv_node = insert_recv_plugin(graph, hidden_states_input.node, mapping.prev_pp_rank)
                pipeline_input_node.replace_all_uses_with(recv_node)
                if (
                    len(users := list(recv_node.users)) == 2
                    and (common_descendant := find_closest_common_descendant(users[0], users[1]))
                    and (binary := BinaryElementwise.specialize_from(common_descendant))
                ):
                    # Note: Some models have squeeze and unsqueeze operations for broadcasting.
                    # In that case, it might occur the mismatch of the input shapes of some binary nodes.
                    # This function resolves the mismatch by unsqueezing the input node if necessary.
                    resolve_unmatched_input_shapes(binary)
            elif (
                not mapping.is_last_pp_rank()
                and gpt_attn_plugin.layer_idx == layers_to_be_parallelized[-1]
                and (next_gpt_attn_plugin_node := find_next_gpt_attn_plugin_node(node))
                and (pipeline_output_node := find_input_node_of_pipeline(next_gpt_attn_plugin_node))
                and (graph_output_node := find_output_node(graph))
            ):
                send_node = insert_send_plugin(graph, pipeline_output_node, mapping.next_pp_rank)
                graph_output_node.replace_all_uses_with(send_node)

            gpt_attn_plugin.layer_idx = gpt_attn_plugin.layer_idx - layers_to_be_parallelized[0]
            gpt_attn_plugin.layer_idx_in_cache_pool = gpt_attn_plugin.layer_idx
            overall_modified = True

        if overall_modified:
            cleanup(graph_module)
            unused_placeholders = [node for node in graph.find_nodes(op="placeholder") if len(node.users) == 0]
            for placeholder in unused_placeholders:
                graph.erase_node(placeholder)

        return PassResult(graph_module=graph_module, modified=False)


def get_num_gpt_attn_plugin_nodes(graph: Graph) -> int:
    """Get the number of GPTAttentionPlugin nodes in the graph.

    Args:
        graph (Graph): The graph to get the number of GPTAttentionPlugin layers from

    Returns:
        int: The number of GPTAttentionPlugin layers in the graph
    """
    return sum(1 for node in graph.nodes if isinstance(node.target, GPTAttentionPlugin))


def find_next_gpt_attn_plugin_node(node: Node) -> Node | None:
    """Find the next GPTAttentionPlugin node in the graph.

    Args:
        node (Node): The node to find the next GPTAttentionPlugin node from

    Returns:
        Node | None: The next GPTAttentionPlugin node in the graph or None if no such node is found
    """
    visited: set[Node] = set()
    q: list[Node] = list(node.users)
    while q:
        current = q.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if isinstance(current.target, GPTAttentionPlugin):
            return current
        if current.users:
            q.extend(current.users)

    return None


def find_input_ids_node(graph: Graph) -> Node:
    """Find the logits node in the graph.

    Args:
        graph (Graph): The graph to find the logits node from

    Returns:
        Node: The logits node in the graph
    """
    for input_node in graph.find_nodes(op="placeholder"):
        if input_node.name == "input_ids":
            return input_node
    raise RuntimeError("No input_ids placeholder found in the graph")


def find_input_node_of_pipeline(node: Node) -> Node | None:
    """Find the input node of the pipeline.

    Args:
        node (Node): The node to find the input node from

    Returns:
        Node | None: The input node in the pipeline or None if no such node is found
    """

    def find_nearest_add_node() -> Node | None:
        """Find the nearest succeeding add node in the graph.

        Returns:
            Node | None: The nearest add node in the graph or None if no such node is found
        """
        visited: set[Node] = set()
        queue: list[Node] = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if (add := AddTensorTensor.specialize_from(current)) and add.this.op == add.other.op == "call_function":
                return current
            if current.users:
                queue.extend(current.users)

        return None

    if not (expected_output_node := find_nearest_add_node()):
        return None
    visited: set[Node] = set()
    queue: list[Node] = [node]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if len(current.users) == 2 and expected_output_node in current.users:
            return current
        for input_node in current.all_input_nodes:
            queue.append(input_node)

    return None


def insert_send_plugin(graph: Graph, to: Node, tgt_rank: int) -> Node:
    """Insert a send plugin node into the graph.

    Args:
        graph (Graph): The graph to insert the send plugin node into
        to (Node): The node to insert the send plugin node after
        tgt_rank (int): The rank that receives the tensor

    Returns:
        Node: The send plugin node
    """
    assert (to_val := get_val(to, torch.Tensor)) is not None, f"Failed to get tensor value from {to.format_node()}"
    send_plugin = SendPlugin(tgt_rank=tgt_rank, type_id=DataType(dtype=to_val.dtype).to(trt.DataType))
    with graph.inserting_after(to):
        send = graph.call_function(send_plugin, (to,))
        propagate_metadata_from(to, to=send)
    return send


def insert_recv_plugin(graph: Graph, to: Node, src_rank: int) -> Node:
    """Insert a recv plugin node into the graph.

    Args:
        graph (Graph): The graph to insert the recv plugin node into
        to (Node): The node to insert the recv plugin node after
        src_rank (int): The rank that sends the tensor

    Returns:
        Node: The recv plugin node
    """
    assert (to_val := get_val(to, torch.Tensor)) is not None, f"Failed to get tensor value from {to.format_node()}"
    recv_plugin = RecvPlugin(src_rank=src_rank, type_id=DataType(dtype=to_val.dtype).to(trt.DataType))
    with graph.inserting_after(to):
        recv = graph.call_function(recv_plugin, (to,))
        propagate_metadata_from(to, to=recv)
    return recv


def find_output_node(graph: Graph) -> Node:
    """Find the output node in the graph.

    Args:
        graph (Graph): The graph to find the output node from

    Returns:
        Node: The output node in the graph
    """
    output_node = graph.find_nodes(op="output")[0]
    assert isinstance(input_args := output_node.args[0], tuple)
    return input_args[0]


def resolve_unmatched_input_shapes(binary: BinaryElementwise) -> None:
    """Resolve the unmatched input shapes of the binary node.

    This function resolves only the case where one of the inputs needs to be unsqueezed.

    Example:
        binary.this.shape = (dim0, dim1)
        binary.other.shape = (1, dim0, dim1)

        After resolution:
        binary.this.shape = (1, dim0, dim1)
        binary.other.shape = (1, dim0, dim1)

    Args:
        binary (BinaryElementwise): The binary node to resolve the unmatched input shapes of
    """

    def need_to_unsqueeze(shape0: tuple[int | torch.SymInt, ...], shape1: tuple[int | torch.SymInt, ...]) -> bool:
        if (
            len(shape1) - len(shape0) == 1
            and isinstance(shape1[0], int)
            and shape1[0] == 1
            and all(lhs_dim == rhs_dim for lhs_dim, rhs_dim in zip(shape0, shape1[1:]))
        ):
            return True
        return False

    if (
        (lhs_val := get_val(binary.this, FakeTensor)) is not None
        and (rhs_val := get_val(binary.other, FakeTensor)) is not None
        and (
            need_to_unsqueeze(lhs_val.shape, rhs_val.shape)
            if len(lhs_val.shape) < len(rhs_val.shape)
            else need_to_unsqueeze(rhs_val.shape, lhs_val.shape)
        )
    ):
        target_node = binary.this if len(lhs_val.shape) < len(rhs_val.shape) else binary.other
        with target_node.graph.inserting_after(target_node):
            unsqueeze = Unsqueeze.create(target_node.graph, target_node, 0)
            target_node.replace_all_uses_with(unsqueeze.node, delete_user_cb=lambda user: user is not unsqueeze.node)
