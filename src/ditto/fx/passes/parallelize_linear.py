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
from pydantic import Field
from tensorrt_llm.functional import AllReduceConfig, AllReduceFusionOp, AllReduceStrategy
from torch.fx import Graph, GraphModule, Node

from ...configs import TRTLLMMapping
from ...types import DataType
from ..nodes import GetAttr, Permute, Reshape
from ..subgraphs import Linear
from ..targets import AllGatherPlugin, AllReducePlugin, AllReducePluginInputs, GPTAttentionPlugin
from ..utils import forget_all_descendant_fake_tensors, get_val
from .infra import (
    GraphOptimizationPass,
    PassResult,
    propagate_metadata_from,
)
from .propagate_tensor_parallelism import TensorParallelType


# TODO: Change ParallelizeLinear to inherit from NodewiseOptimization instead of GraphOptimizationPass
# This will allow processing nodes individually rather than the whole graph at once
class ParallelizeLinear(GraphOptimizationPass):
    """Parallelize linear nodes in the graph (Tensor Parallelism).

    This pass must be run after PropagateTensorParallelism pass.

    Attributes:
        mapping (TRTLLMMapping): The mapping of the model
    """

    mapping: TRTLLMMapping = Field(frozen=True)

    def call(self, graph_module: GraphModule) -> PassResult:
        first_node_rewritten: Node | None = None
        overall_modified = False
        graph = graph_module.graph
        lm_head = Linear.find_last(graph)
        for node in graph.nodes:
            if not (linear := Linear.configure_from(node)):
                continue

            tp_type = node.meta.get("tp_type", TensorParallelType.NONE)
            if tp_type == TensorParallelType.COLUMN:
                gpt_attn_plugin = get_gpt_plugin(linear)
                self.parallelize_column_linear(
                    linear,
                    gather_output=linear == lm_head,
                    is_qkv=gpt_attn_plugin is not None,
                    q_dim=-1
                    if gpt_attn_plugin is None
                    else self.mapping.tp_size * gpt_attn_plugin.num_heads * gpt_attn_plugin.head_size,
                    kv_dim=-1
                    if gpt_attn_plugin is None
                    else self.mapping.tp_size * gpt_attn_plugin.num_kv_heads * gpt_attn_plugin.head_size,
                )
            elif tp_type == TensorParallelType.ROW:
                self.parallelize_row_linear(linear)

            if first_node_rewritten is None:
                first_node_rewritten = node

            overall_modified = True

        if first_node_rewritten is not None:
            forget_all_descendant_fake_tensors(first_node_rewritten)

        return PassResult(graph_module=graph_module, modified=overall_modified, require_fake_tensor_prop=True)

    def get_name_of_attr(self, from_name: str) -> str:
        """Create the name of the attribute with the tensor parallel rank.

        Args:
            from_name (str): The name of the attribute

        Returns:
            str: The name of the attribute with the tensor parallel rank
        """
        return f"{from_name}_rank{self.mapping.tp_rank}"

    # pylint: disable-next=too-many-locals,too-many-statements
    def parallelize_column_linear(
        self,
        linear: Linear,
        *,
        gather_output: bool = True,
        is_qkv: bool = False,
        q_dim: int = -1,
        kv_dim: int = -1,
    ) -> None:
        """Parallelize the linear subgraph in column direction.

        Args:
            linear (Linear): The linear subgraph to be parallelized
            gather_output (bool, optional): Whether to gather the output. Defaults to True.
            is_qkv (bool, optional): Whether the linear subgraph is a QKV linear. Defaults to False.
            q_dim (int, optional): The dimension of the query. Defaults to -1.
            kv_dim (int, optional): The dimension of the key/value. Defaults to -1.
        """
        if not (weight := GetAttr.specialize_from(linear.weight_node)):
            return
        local_out_features = linear.out_features // self.mapping.tp_size
        if is_qkv:
            # qkv weight is already merged, so it needs to be split
            assert (
                q_dim % self.mapping.tp_size == 0 and kv_dim % self.mapping.tp_size == 0
            ), "q_dim and kv_dim must be divisible by tp_size"
            q_slice_size = q_dim // self.mapping.tp_size
            kv_slice_size = kv_dim // self.mapping.tp_size
            if linear.has_transposed_weight:
                q = weight.tensor[:q_dim, :]
                q_slices = [q[q_slice_size * i : q_slice_size * (i + 1), :] for i in range(self.mapping.tp_size)]
                k = weight.tensor[q_dim : q_dim + kv_dim, :]
                k_slices = [k[kv_slice_size * i : kv_slice_size * (i + 1), :] for i in range(self.mapping.tp_size)]
                v = weight.tensor[q_dim + kv_dim :, :]
                v_slices = [v[kv_slice_size * i : kv_slice_size * (i + 1), :] for i in range(self.mapping.tp_size)]

                parallelized_weight_tensor = torch.cat(
                    [q_slices[self.mapping.tp_rank], k_slices[self.mapping.tp_rank], v_slices[self.mapping.tp_rank]],
                    dim=0,
                )
            else:
                q = weight.tensor[:, :q_dim]
                q_slices = [q[:, q_slice_size * i : q_slice_size * (i + 1)] for i in range(self.mapping.tp_size)]
                k = weight.tensor[:, q_dim : q_dim + kv_dim]
                k_slices = [k[:, kv_slice_size * i : kv_slice_size * (i + 1)] for i in range(self.mapping.tp_size)]
                v = weight.tensor[:, q_dim + kv_dim :]
                v_slices = [v[:, kv_slice_size * i : kv_slice_size * (i + 1)] for i in range(self.mapping.tp_size)]

                parallelized_weight_tensor = torch.cat(
                    [q_slices[self.mapping.tp_rank], k_slices[self.mapping.tp_rank], v_slices[self.mapping.tp_rank]],
                    dim=1,
                )
        else:
            slice_size = local_out_features
            if linear.has_transposed_weight:
                parallelized_weight_tensor = weight.tensor[
                    slice_size * self.mapping.tp_rank : slice_size * (self.mapping.tp_rank + 1), :
                ]
            else:
                parallelized_weight_tensor = weight.tensor[
                    :, slice_size * self.mapping.tp_rank : slice_size * (self.mapping.tp_rank + 1)
                ]

        assert (
            parallelized_weight_tensor.shape
            == (
                local_out_features,
                linear.in_features,
            )
            if linear.has_transposed_weight
            else (
                linear.in_features,
                local_out_features,
            )
        ), "unexpected shape of parallelized weight"

        graph = weight.node.graph
        with graph.inserting_before(weight.node):
            parallelized_weight = GetAttr.create(
                graph, self.get_name_of_attr(weight.target), parallelized_weight_tensor
            )
            propagate_metadata_from(weight, to=parallelized_weight)
        weight.node.replace_all_uses_with(parallelized_weight.node)
        linear.mm.other = parallelized_weight.node

        if (
            linear.bias_node is not None
            and linear.add is not None
            and (bias := GetAttr.specialize_from(linear.bias_node))
        ):
            if is_qkv:
                q_bias = bias.tensor[:q_dim]
                q_bias_slices = [q_bias[q_slice_size * i : q_slice_size * (i + 1)] for i in range(self.mapping.tp_size)]
                k_bias = bias.tensor[q_dim : q_dim + kv_dim]
                k_bias_slices = [
                    k_bias[kv_slice_size * i : kv_slice_size * (i + 1)] for i in range(self.mapping.tp_size)
                ]
                v_bias = bias.tensor[q_dim + kv_dim :]
                v_bias_slices = [
                    v_bias[kv_slice_size * i : kv_slice_size * (i + 1)] for i in range(self.mapping.tp_size)
                ]

                parallelized_bias_tensor = torch.cat(
                    [
                        q_bias_slices[self.mapping.tp_rank],
                        k_bias_slices[self.mapping.tp_rank],
                        v_bias_slices[self.mapping.tp_rank],
                    ],
                )
            else:
                slice_size = local_out_features
                parallelized_bias_tensor = bias.tensor[
                    slice_size * self.mapping.tp_rank : slice_size * (self.mapping.tp_rank + 1)
                ]

            with graph.inserting_before(bias.node):
                parallelized_bias = GetAttr.create(graph, self.get_name_of_attr(bias.target), parallelized_bias_tensor)
                propagate_metadata_from(bias, to=parallelized_bias)
            bias.node.replace_all_uses_with(parallelized_bias.node)
            linear.add.other = parallelized_bias.node

        if gather_output:
            insert_allgather_plugin(graph, linear.output_node, self.mapping.tp_group)

    def parallelize_row_linear(
        self,
        linear: Linear,
        *,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        config: AllReduceConfig = AllReduceConfig(0),  # noqa: B008
        fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE,
        eps: float = 1e-5,
    ) -> None:
        """Parallelize the linear subgraph in row direction.

        Args:
            linear (Linear): The linear subgraph to be parallelized
            strategy (AllReduceStrategy, optional): The strategy of the allreduce plugin.
                Defaults to AllReduceStrategy.AUTO.
            config (AllReduceConfig, optional): The config of the allreduce plugin. Defaults to AllReduceConfig(0).
            fusion_op (AllReduceFusionOp, optional): The fusion operation of the allreduce plugin.
                Defaults to AllReduceFusionOp.NONE.
            eps (float, optional): The epsilon value of the allreduce plugin. Defaults to 1e-5.
        """
        if not (weight := GetAttr.specialize_from(linear.weight_node)):
            return
        local_in_features = linear.in_features // self.mapping.tp_size
        if linear.has_transposed_weight:
            parallelized_weight_tensor = weight.tensor[
                :, local_in_features * self.mapping.tp_rank : local_in_features * (self.mapping.tp_rank + 1)
            ]
        else:
            parallelized_weight_tensor = weight.tensor[
                local_in_features * self.mapping.tp_rank : local_in_features * (self.mapping.tp_rank + 1), :
            ]

        assert (
            parallelized_weight_tensor.shape
            == (
                linear.out_features,
                local_in_features,
            )
            if linear.has_transposed_weight
            else (
                local_in_features,
                linear.out_features,
            )
        ), "unexpected shape of parallelized weight"

        graph = weight.node.graph
        with graph.inserting_before(weight.node):
            parallelized_weight = GetAttr.create(
                graph, self.get_name_of_attr(weight.target), parallelized_weight_tensor
            )
            propagate_metadata_from(weight, to=parallelized_weight)
        weight.node.replace_all_uses_with(parallelized_weight.node)
        linear.mm.other = parallelized_weight.node

        insert_allreduce_plugin(
            graph,
            linear.mm.node,
            self.mapping.tp_group,
            strategy=strategy,
            config=config,
            fusion_op=fusion_op,
            eps=eps,
        )


# [TODO] This function is only supported for 2D tensor, it should be extended to support arbitrary dimensions
# pylint: disable-next=unused-argument
def insert_allgather_plugin(graph: Graph, to: Node, group: list[int], gather_dim: int = 0) -> None:
    """Insert an allgather plugin node into the graph.

    Args:
        graph (Graph): The graph to insert the allgather plugin node into
        to (Node): The source node to be replaced
        group (list[int]): The group of the allgather plugin
        gather_dim (int, optional): The dimension to gather. Defaults to 0.
    """
    group_size = len(group)
    input_tensor = get_val(to, torch.Tensor)
    # if gather_dim < 0:
    #     assert gather_dim == -1, "gather_dim must be -1 when gather_dim is lower than 0"
    #     gather_dim = input_tensor.ndim - 1
    allgather_plugin = AllGatherPlugin(group=group, type_id=DataType(input_tensor.dtype).to(trt.DataType))
    with graph.inserting_after(to):
        allgather = graph.call_function(allgather_plugin, (to,))
    with graph.inserting_after(allgather):
        reshape_1 = Reshape.create(graph, allgather, (group_size, -1, input_tensor.shape[-1] // group_size))
    with graph.inserting_after(reshape_1.node):
        permute = Permute.create(graph, reshape_1, (1, 0, 2))
    with graph.inserting_after(permute.node):
        reshape_2 = Reshape.create(graph, permute, (-1, input_tensor.shape[-1]))

    to.replace_all_uses_with(reshape_2.node, delete_user_cb=lambda user: user not in (reshape_2.node, allgather))


def insert_allreduce_plugin(
    graph: Graph,
    to: Node,
    group: list[int],
    *,
    strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    config: AllReduceConfig = AllReduceConfig(0),  # noqa: B008
    fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE,
    eps: float = 1e-5,
) -> Node:
    """Insert an allreduce plugin node into the graph.

    Args:
        graph (Graph): The graph to insert the allreduce plugin node into
        to (Node): The source node to be replaced
        group (list[int]): The group of the allreduce plugin
        strategy (AllReduceStrategy, optional): The strategy of the allreduce plugin.
            Defaults to AllReduceStrategy.AUTO.
        config (AllReduceConfig, optional): The config of the allreduce plugin. Defaults to AllReduceConfig(0).
        fusion_op (AllReduceFusionOp, optional): The fusion operation of the allreduce plugin.
            Defaults to AllReduceFusionOp.NONE.
        eps (float, optional): The epsilon value of the allreduce plugin. Defaults to 1e-5.

    Returns:
        Node: The allreduce plugin node
    """
    assert (to_val := get_val(to, torch.Tensor)) is not None, f"Failed to get tensor value from {to.format_node()}"
    allreduce_plugin = AllReducePlugin(
        group=group,
        type_id=DataType(dtype=to_val.dtype).to(trt.DataType),
        strategy=strategy,
        config=config,
        fusion_op=fusion_op,
        eps=eps,
    )
    plugin_inputs = AllReducePluginInputs.find_from(graph, allreduce_plugin)

    with graph.inserting_after(to):
        allreduce = graph.call_function(
            allreduce_plugin,
            (to,),
            plugin_inputs.model_dump(),
        )
    to.replace_all_uses_with(allreduce, delete_user_cb=lambda user: user is not allreduce)


def get_gpt_plugin(linear: Linear) -> GPTAttentionPlugin | None:
    """Get the GPTAttentionPlugin from the linear node.

    Args:
        linear (Linear): The linear node to get the GPTAttentionPlugin from

    Returns:
        GPTAttentionPlugin | None: The GPTAttentionPlugin from the linear node
    """
    return (
        target
        if len(users := list(linear.output_node.users)) == 1
        and isinstance(target := users[0].target, GPTAttentionPlugin)
        else None
    )
