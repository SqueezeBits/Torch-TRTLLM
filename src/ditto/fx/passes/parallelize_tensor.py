from copy import copy
from enum import IntEnum

import tensorrt as trt
import torch
from tensorrt_llm.functional import AllReduceConfig, AllReduceFusionOp, AllReduceStrategy
from torch.fx import Graph, GraphModule, Node

from ...configs.trtllm.pretrained import TRTLLMMapping
from ...types import DataType
from ..nodes import GetAttr, Permute, Reshape
from ..subgraphs import TokenEmbedding
from ..subgraphs.linear import Linear
from ..targets import AllGatherPlugin, AllReducePlugin, AllReducePluginInputs, GPTAttentionPlugin
from ..utils import forget_all_descendant_fake_tensors, get_val
from .infra import (
    GraphOptimizationPass,
    PassResult,
    inject_stack_trace_from,
)


class TensorParallelType(IntEnum):
    """Tensor parallel type.

    Attributes:
        NONE (int): The tensor parallel type is none
        COLUMN (int): The tensor parallel type is column
    """

    NONE = 0
    COLUMN = 1

    def __str__(self) -> str:
        return self.name


# TODO: Change ParallelizeTensor to inherit from NodewiseOptimization instead of GraphOptimizationPass
# This will allow processing nodes individually rather than the whole graph at once
class ParallelizeTensor(GraphOptimizationPass):
    """Parallelize tensor in the graph (Tensor Parallelism).

    This pass is designed to parallelize the tensor in the graph.
    It propagates the tensor parallel type in the path of current node,
    and parallelizes the node that meet the conditions according to the following rules:
    - The possible values of tensor parallel type are "none" and "column"
    - If the current node is a linear node and the previous tensor parallel type is "none",
      it will parallelize the node in column direction and set the tensor parallel type of this node to "column".
    - If the current node is a linear node and the previous tensor parallel type is "column",
      it will parallelize the node in row direction and set the tensor parallel type of this node to "none".
      The linear node that is parallelized in row direction has the reduce operation.
    - If the current node is the lm_head node,
      it will parallelize the node in column direction with allgather operation.
    - If the current node is not a linear node, it just propagates the tensor parallel type of the previous node.

    Attributes:
        mapping (TRTLLMMapping): The mapping of the model
    """

    mapping: TRTLLMMapping

    def call(self, graph_module: GraphModule) -> PassResult:
        vocab_size: int | None = None
        hidden_size: int | None = None
        attention_head_size: int | None = None
        num_attention_heads: int | None = None
        num_attention_kv_heads: int | None = None
        first_node_rewritten: Node | None = None

        overall_modified = False
        for node in graph_module.graph.nodes:
            modified = False
            tp_type = get_previous_tp_type(node)

            if (token_embedding := TokenEmbedding.configure_from(node)) is not None:
                vocab_size = token_embedding.vocab_size
                hidden_size = token_embedding.hidden_size
            elif (linear := Linear.configure_from(node)) is not None and (
                weight := GetAttr.specialize_from(linear.weight_node)
            ) is not None:
                assert hidden_size is not None and vocab_size is not None, "hidden_size and vocab_size must be set"
                in_features = weight.tensor.shape[1 if linear.has_transposed_weight else 0]
                out_features = weight.tensor.shape[0 if linear.has_transposed_weight else 1]
                is_lm_head = vocab_size in (in_features, out_features)
                if len(users := list(linear.output_node.users)) == 1 and isinstance(
                    users[0].target, GPTAttentionPlugin
                ):
                    gpt_attn_plugin = users[0].target
                    if not (
                        attention_head_size is not None
                        and num_attention_heads is not None
                        and num_attention_kv_heads is not None
                    ):
                        # get attention's parameters
                        attention_head_size = gpt_attn_plugin.head_size
                        num_attention_heads = gpt_attn_plugin.num_heads // self.mapping.tp_size
                        num_attention_kv_heads = (
                            gpt_attn_plugin.num_kv_heads + self.mapping.tp_size - 1
                        ) // self.mapping.tp_size

                    if tp_type == TensorParallelType.NONE:
                        self.parallelize_column_linear(
                            graph_module.graph,
                            linear,
                            in_features,
                            out_features,
                            gather_output=False,
                            is_qkv=True,
                            q_dim=self.mapping.tp_size * num_attention_heads * attention_head_size,
                            kv_dim=self.mapping.tp_size * num_attention_kv_heads * attention_head_size,
                        )
                        tp_type = TensorParallelType.COLUMN
                    else:
                        raise RuntimeError("QKV linear is always parallelized in column direction")
                elif is_lm_head:
                    self.parallelize_column_linear(
                        graph_module.graph,
                        linear,
                        in_features,
                        out_features,
                        gather_output=True,
                    )
                    tp_type = TensorParallelType.NONE
                else:
                    if tp_type == TensorParallelType.NONE:
                        self.parallelize_column_linear(
                            graph_module.graph,
                            linear,
                            in_features,
                            out_features,
                            gather_output=False,
                        )
                        tp_type = TensorParallelType.COLUMN
                    elif tp_type == TensorParallelType.COLUMN:
                        self.parallelize_row_linear(
                            graph_module.graph,
                            linear,
                            in_features,
                            out_features,
                        )
                        tp_type = TensorParallelType.NONE
                modified = True
            elif isinstance(node.target, GPTAttentionPlugin):
                node.target = copy(node.target)
                node.target.num_heads = num_attention_heads
                node.target.num_kv_heads = num_attention_kv_heads
                node.target.tp_size = self.mapping.tp_size
                node.target.tp_rank = self.mapping.tp_rank
                modified = True

            node.meta["tp_type"] = tp_type
            if modified and first_node_rewritten is None:
                first_node_rewritten = node

            overall_modified = overall_modified or modified

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
        graph: Graph,
        linear: Linear,
        in_features: int,
        out_features: int,
        *,
        gather_output: bool = True,
        is_qkv: bool = False,
        q_dim: int = -1,
        kv_dim: int = -1,
    ) -> None:
        """Parallelize the linear subgraph in column direction.

        Args:
            graph (Graph): The graph to insert the parallelized linear subgraph into
            linear (Linear): The linear subgraph to be parallelized
            in_features (int): The input feature size
            out_features (int): The output feature size
            gather_output (bool, optional): Whether to gather the output. Defaults to True.
            is_qkv (bool, optional): Whether the linear subgraph is a QKV linear. Defaults to False.
            q_dim (int, optional): The dimension of the query. Defaults to -1.
            kv_dim (int, optional): The dimension of the key/value. Defaults to -1.
        """
        weight = GetAttr.specialize_from(linear.weight_node)
        local_out_features = out_features // self.mapping.tp_size
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

        assert parallelized_weight_tensor.ndim == 2 and tuple(parallelized_weight_tensor.shape) == (
            local_out_features if linear.has_transposed_weight else in_features,
            in_features if linear.has_transposed_weight else local_out_features,
        ), "unexpected shape of parallelized weight"

        with graph.inserting_before(weight.node):
            parallelized_weight = GetAttr.create(
                graph, self.get_name_of_attr(weight.target), parallelized_weight_tensor
            )
            inject_stack_trace_from(weight, to=parallelized_weight)
        weight.node.replace_all_uses_with(parallelized_weight.node)
        linear.mm.other = parallelized_weight.node

        if linear.bias_node is not None:
            bias = GetAttr.specialize_from(linear.bias_node)
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
                inject_stack_trace_from(bias, to=parallelized_bias)
            bias.node.replace_all_uses_with(parallelized_bias.node)
            linear.add.other = parallelized_bias.node

        if gather_output:
            insert_allgather_plugin(graph, linear.output_node, self.mapping.tp_group)

    def parallelize_row_linear(
        self,
        graph: Graph,
        linear: Linear,
        in_features: int,
        out_features: int,
        *,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        config: AllReduceConfig = AllReduceConfig(0),
        fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE,
        eps: float = 1e-5,
    ) -> None:
        """Parallelize the linear subgraph in row direction.

        Args:
            graph (Graph): The graph to insert the parallelized linear subgraph into
            linear (Linear): The linear subgraph to be parallelized
            in_features (int): The input feature size
            out_features (int): The output feature size
            strategy (AllReduceStrategy, optional): The strategy of the allreduce plugin.
                Defaults to AllReduceStrategy.AUTO.
            config (AllReduceConfig, optional): The config of the allreduce plugin. Defaults to AllReduceConfig(0).
            fusion_op (AllReduceFusionOp, optional): The fusion operation of the allreduce plugin.
                Defaults to AllReduceFusionOp.NONE.
            eps (float, optional): The epsilon value of the allreduce plugin. Defaults to 1e-5.
        """
        weight = GetAttr.specialize_from(linear.weight_node)
        local_in_features = in_features // self.mapping.tp_size
        slice_size = local_in_features
        if linear.has_transposed_weight:
            parallelized_weight_tensor = weight.tensor[
                :, slice_size * self.mapping.tp_rank : slice_size * (self.mapping.tp_rank + 1)
            ]
        else:
            parallelized_weight_tensor = weight.tensor[
                slice_size * self.mapping.tp_rank : slice_size * (self.mapping.tp_rank + 1), :
            ]

        assert parallelized_weight_tensor.ndim == 2 and tuple(parallelized_weight_tensor.shape) == (
            out_features if linear.has_transposed_weight else local_in_features,
            local_in_features if linear.has_transposed_weight else out_features,
        ), "unexpected shape of parallelized weight"

        with graph.inserting_before(weight.node):
            parallelized_weight = GetAttr.create(
                graph, self.get_name_of_attr(weight.target), parallelized_weight_tensor
            )
            inject_stack_trace_from(weight, to=parallelized_weight)
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
    config: AllReduceConfig = AllReduceConfig(0),
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
    allreduce_plugin = AllReducePlugin(
        group=group,
        type_id=DataType(dtype=get_val(to).dtype).to(trt.DataType),
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


def get_previous_tp_type(node: Node) -> TensorParallelType:
    """Get the previous parallel linear type of the node.

    Args:
        node (Node): The node to get the previous parallel linear type from

    Returns:
        TensorParallelType: The previous parallel linear type
    """
    if len(node.all_input_nodes) == 0:
        return TensorParallelType.NONE
    prev_tp_types: list[TensorParallelType] = []
    for prev_node in node.all_input_nodes:
        prev_tp_types.append(prev_node.meta.get("tp_type", TensorParallelType.NONE))
    return TensorParallelType.COLUMN if TensorParallelType.COLUMN in prev_tp_types else TensorParallelType.NONE
