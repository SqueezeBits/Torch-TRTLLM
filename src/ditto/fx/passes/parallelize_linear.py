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
from ..nodes import Embedding, Expand, GetAttr, Permute, Reshape, Slice
from ..subgraphs import FusedLinear, Linear
from ..targets import AllGatherPlugin, AllReducePlugin, AllReducePluginInputs
from ..utils import forget_all_descendant_fake_tensors, get_val
from .infra import (
    GraphOptimizationPass,
    PassResult,
    propagate_metadata_from,
)
from .propagate_tensor_parallelism import TensorParallelType


class ParallelizeLinear(GraphOptimizationPass):
    """Parallelize linear nodes in the graph (Tensor Parallelism).

    This pass must be run after PropagateTensorParallelism pass.

    Attributes:
        mapping (TRTLLMMapping): The mapping of the model
    """

    mapping: TRTLLMMapping = Field(frozen=True)

    def call(self, graph_module: GraphModule) -> PassResult:
        if self.mapping.tp_size == 1:
            return PassResult(graph_module=graph_module, modified=False)

        overall_modified = False
        graph = graph_module.graph
        lm_head = Linear.find_last(graph)
        for node in graph.nodes:
            if not (
                linear := (
                    fused_linear.linear
                    if (fused_linear := FusedLinear.configure_from(node))
                    else Linear.configure_from(node)
                )
            ):
                continue

            tp_type = node.meta.get("tp_type", TensorParallelType.NONE)
            if tp_type == TensorParallelType.COLUMN:
                slice_dim: list[int] | None = None
                if fused_linear:
                    slice_dim = [0]
                    for slice_node in fused_linear.slices:
                        assert isinstance(slice_node.end, int)
                        slice_dim.append(slice_node.end)
                        parallelize_slice(slice_node, self.mapping)

                parallelize_column_linear(linear, self.mapping, slice_dim=slice_dim, gather_output=linear == lm_head)
            elif tp_type == TensorParallelType.ROW:
                parallelize_row_linear(linear, self.mapping)

            overall_modified = True

        if overall_modified:
            # resolve reformatting nodes
            embedding: Embedding | None = None
            for node in graph.nodes:
                if node.meta.get("tp_type", TensorParallelType.NONE) == TensorParallelType.NONE:
                    embedding = embedding or Embedding.specialize_from(node)
                    continue
                parallelize_reformat(node, self.mapping)

            assert embedding is not None, "embedding node not found"
            forget_all_descendant_fake_tensors(embedding.node)

        return PassResult(
            graph_module=graph_module, modified=overall_modified, require_fake_tensor_prop=overall_modified
        )


def get_name_of_attr(from_name: str, tp_rank: int) -> str:
    """Create the name of the attribute with the tensor parallel rank.

    Args:
        from_name (str): The name of the attribute to append the rank to
        tp_rank (int): The tensor parallel rank to append

    Returns:
        str: The name of the attribute with the tensor parallel rank
    """
    return f"{from_name}_rank{tp_rank}"


def parallelize_column_linear(
    linear: Linear,
    mapping: TRTLLMMapping,
    *,
    slice_dim: list[int] | None = None,
    gather_output: bool = True,
    inplace: bool = False,
) -> None:
    """Parallelize the linear subgraph in column direction.

    Args:
        linear (Linear): The linear subgraph to be parallelized
        mapping (TRTLLMMapping): The tensor parallel mapping configuration
        slice_dim (list[int] | None, optional): The dimension to slice. Defaults to None.
        gather_output (bool, optional): Whether to gather the output. Defaults to True.
        inplace (bool, optional): Whether to modify the linear subgraph inplace. Defaults to False.
    """
    graph = linear.mm.node.graph

    if (dequantize := linear.weight_dequantize_node) is not None:
        input_nodes: list[Node] = []
        for dequant_input_node in linear.mm.other.all_input_nodes:
            assert (
                tensor := GetAttr.specialize_from(dequant_input_node)
            ), "dequantize's input node is not specialized to GetAttr"
            with graph.inserting_before(tensor.node):
                parallelized_weight = GetAttr.create(
                    graph,
                    get_name_of_attr(tensor.target, mapping.tp_rank),
                    tensor.tensor
                    if len(tensor.tensor.shape) == 1 or tensor.tensor.numel() == 1
                    else parallelize_2d_tensor(
                        tensor.tensor,
                        tp_size=mapping.tp_size,
                        tp_rank=mapping.tp_rank,
                        is_column=True,
                        is_transposed=False,
                    ),
                )
                propagate_metadata_from(tensor, to=parallelized_weight)
            dequant_input_node.replace_all_uses_with(parallelized_weight.node)
            input_nodes.append(parallelized_weight.node)
        parallelized_dequantize = dequantize.target.model_copy(
            update={
                "output_shape": torch.Size(
                    [dequantize.target.output_shape[0], dequantize.target.output_shape[1] // mapping.tp_size]
                )
            }
        )
        with graph.inserting_before(linear.mm.other):
            parallelized_dequantize_node = graph.call_function(parallelized_dequantize, tuple(input_nodes))
            propagate_metadata_from(linear.mm.other, to=parallelized_dequantize_node)
        linear.mm.other.replace_all_uses_with(parallelized_dequantize_node)
        if inplace:
            linear.mm.other = parallelized_dequantize_node
    else:
        assert (
            weight := GetAttr.specialize_from(linear.weight_node)
        ) is not None, "weight node is not specialized to GetAttr"
        weight_tensor = weight.tensor
        if linear.has_transposed_weight:
            weight_tensor = weight_tensor.T

        with linear.mm.node.graph.inserting_before(weight.node):
            parallelized_weight = GetAttr.create(
                linear.mm.node.graph,
                get_name_of_attr(weight.target, mapping.tp_rank),
                parallelize_2d_tensor(
                    weight_tensor,
                    tp_size=mapping.tp_size,
                    tp_rank=mapping.tp_rank,
                    is_column=True,
                    is_transposed=linear.has_transposed_weight,
                    slice_dim=slice_dim,
                ),
            )
            propagate_metadata_from(weight, to=parallelized_weight)

        weight.node.replace_all_uses_with(parallelized_weight.node)
        if inplace:
            linear.mm.other = parallelized_weight.node

    if linear.bias_node is not None and linear.add is not None and (bias := GetAttr.specialize_from(linear.bias_node)):
        if slice_dim:
            parallelized_bias_tensor = torch.concat(
                [
                    bias.tensor[
                        slice_dim[i]
                        + (slice_dim[i + 1] - slice_dim[i]) // mapping.tp_size * mapping.tp_rank : slice_dim[i]
                        + (slice_dim[i + 1] - slice_dim[i]) // mapping.tp_size * (mapping.tp_rank + 1),
                    ]
                    for i in range(len(slice_dim) - 1)
                ],
                dim=0,
            )
        else:
            local_out_features = linear.out_features // mapping.tp_size
            parallelized_bias_tensor = bias.tensor[
                local_out_features * mapping.tp_rank : local_out_features * (mapping.tp_rank + 1)
            ]

        with graph.inserting_before(bias.node):
            parallelized_bias = GetAttr.create(
                graph, get_name_of_attr(bias.target, mapping.tp_rank), parallelized_bias_tensor
            )
            propagate_metadata_from(bias, to=parallelized_bias)
        bias.node.replace_all_uses_with(parallelized_bias.node)
        if inplace:
            linear.add.other = parallelized_bias.node

    if gather_output:
        insert_allgather_plugin(graph, linear.output_node, mapping.tp_group)


def parallelize_row_linear(
    linear: Linear,
    mapping: TRTLLMMapping,
    *,
    strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    config: AllReduceConfig = AllReduceConfig(0),  # noqa: B008
    fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE,
    eps: float = 1e-5,
) -> None:
    """Parallelize the linear subgraph in row direction.

    Args:
        linear (Linear): The linear subgraph to be parallelized
        mapping (TRTLLMMapping): The tensor parallelism mapping configuration
        strategy (AllReduceStrategy, optional): The strategy of the allreduce plugin.
            Defaults to AllReduceStrategy.AUTO.
        config (AllReduceConfig, optional): The config of the allreduce plugin. Defaults to AllReduceConfig(0).
        fusion_op (AllReduceFusionOp, optional): The fusion operation of the allreduce plugin.
            Defaults to AllReduceFusionOp.NONE.
        eps (float, optional): The epsilon value of the allreduce plugin. Defaults to 1e-5.
    """
    graph = linear.mm.node.graph
    if (dequantize := linear.weight_dequantize_node) is not None:
        input_nodes: list[Node] = []
        for dequant_input_node in linear.mm.other.all_input_nodes:
            assert (
                tensor := GetAttr.specialize_from(dequant_input_node)
            ), "dequantize's input node is not specialized to GetAttr"
            with graph.inserting_before(tensor.node):
                parallelized_weight = GetAttr.create(
                    graph,
                    get_name_of_attr(tensor.target, mapping.tp_rank),
                    tensor.tensor
                    if len(tensor.tensor.shape) == 1 or tensor.tensor.numel() == 1
                    else parallelize_2d_tensor(
                        tensor.tensor,
                        tp_size=mapping.tp_size,
                        tp_rank=mapping.tp_rank,
                        is_column=False,
                        is_transposed=False,
                    ),
                )
                propagate_metadata_from(tensor, to=parallelized_weight)
            dequant_input_node.replace_all_uses_with(parallelized_weight.node)
            input_nodes.append(parallelized_weight.node)
        parallelized_dequantize = dequantize.target.model_copy(
            update={
                "output_shape": torch.Size(
                    [dequantize.target.output_shape[0] // mapping.tp_size, dequantize.target.output_shape[1]]
                )
            }
        )
        with graph.inserting_before(linear.mm.other):
            parallelized_dequantize_node = graph.call_function(parallelized_dequantize, tuple(input_nodes))
            propagate_metadata_from(linear.mm.other, to=parallelized_dequantize_node)
        linear.mm.other.replace_all_uses_with(parallelized_dequantize_node)
    else:
        assert (
            weight := GetAttr.specialize_from(linear.weight_node)
        ) is not None, "weight node is not specialized to GetAttr"
        with graph.inserting_before(weight.node):
            parallelized_weight = GetAttr.create(
                graph,
                get_name_of_attr(weight.target, mapping.tp_rank),
                parallelize_2d_tensor(
                    weight.tensor,
                    tp_size=mapping.tp_size,
                    tp_rank=mapping.tp_rank,
                    is_column=False,
                    is_transposed=linear.has_transposed_weight,
                ),
            )
            propagate_metadata_from(weight, to=parallelized_weight)
        weight.node.replace_all_uses_with(parallelized_weight.node)

    insert_allreduce_plugin(
        graph,
        linear.mm.node,
        mapping.tp_group,
        strategy=strategy,
        config=config,
        fusion_op=fusion_op,
        eps=eps,
    )


def parallelize_slice(slice_node: Slice, mapping: TRTLLMMapping) -> None:
    """Parallelize the slice node.

    Args:
        slice_node (Slice): The slice node to be parallelized
        mapping (TRTLLMMapping): The tensor parallel mapping configuration containing tp_size and tp_rank
    """
    assert isinstance(slice_node.start, int) and isinstance(slice_node.end, int)
    new_start = slice_node.start // mapping.tp_size
    new_end = slice_node.end // mapping.tp_size
    args, kwargs = slice_node.args_kwargs(start=new_start, end=new_end)
    slice_node.node.args = args
    slice_node.node.kwargs = kwargs


def parallelize_reformat(node: Node, mapping: TRTLLMMapping) -> None:
    """Parallelize a reshape or expand node by adjusting the shape for tensor parallelism.

    Args:
        node (Node): The reshape or expand node to parallelize
        mapping (TRTLLMMapping): The tensor parallel mapping configuration containing tp_size and tp_rank
    """
    if (reshape := Reshape.specialize_from(node) or Expand.specialize_from(node)) and -1 in reshape.shape:
        assert reshape.shape.count(-1) == 1, "reshape and expand node must have exactly one dynamic dimension"

        shard_dim_idx = get_shard_dim_idx(reshape)

        new_shape = list(reshape.shape)
        assert isinstance(shard_dim := new_shape[shard_dim_idx], int)
        new_shape[shard_dim_idx] = shard_dim // mapping.tp_size
        args, kwargs = reshape.args_kwargs(shape=new_shape)
        reshape.node.args = args
        reshape.node.kwargs = kwargs


def parallelize_2d_tensor(
    tensor: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    *,
    is_column: bool,
    is_transposed: bool = False,
    slice_dim: list[int] | None = None,
) -> torch.Tensor:
    """Parallelize the 2D tensor in the column or row direction.

    Args:
        tensor (torch.Tensor): The tensor to be parallelized
        tp_size (int): The size of the tensor parallel
        tp_rank (int): The rank of the tensor parallel
        is_column (bool): Whether the tensor is parallelized in the column direction.
        is_transposed (bool, optional): Whether the tensor is transposed. Defaults to False.
        slice_dim (list[int], optional): The dimension to slice. Defaults to None.
    """
    if not is_column ^ is_transposed:
        tensor = tensor.T
    local_features = tensor.shape[1] // tp_size

    if slice_dim:
        parallelized_tensor = torch.concat(
            [
                tensor[
                    :,
                    slice_dim[i]
                    + (slice_dim[i + 1] - slice_dim[i]) // tp_size * tp_rank : slice_dim[i]
                    + (slice_dim[i + 1] - slice_dim[i]) // tp_size * (tp_rank + 1),
                ]
                for i in range(len(slice_dim) - 1)
            ],
            dim=1,
        )
    else:
        parallelized_tensor = tensor[:, local_features * tp_rank : local_features * (tp_rank + 1)]

    assert parallelized_tensor.shape == (
        tensor.shape[0],
        local_features,
    ), "unexpected shape of parallelized tensor"

    return parallelized_tensor if is_column ^ is_transposed else parallelized_tensor.T


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
    assert isinstance(input_tensor := get_val(to), torch.Tensor), "not found tensor value from the node"
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
    eps: float = 1e-5,  # TODO: default eps value should be 1e-6.
    affine: bool = False,
) -> None:
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
        affine (bool, optional): Whether to apply affine transformation to the tensor. Defaults to False.
    """
    assert (to_val := get_val(to, torch.Tensor)) is not None, f"Failed to get tensor value from {to.format_node()}"
    allreduce_plugin = AllReducePlugin(
        group=group,
        type_id=DataType(dtype=to_val.dtype).to(trt.DataType),
        strategy=strategy,
        config=config,
        fusion_op=fusion_op,
        eps=eps,
        affine=affine,
    )
    plugin_inputs = AllReducePluginInputs.find_from(graph, allreduce_plugin)

    with graph.inserting_after(to):
        allreduce = graph.call_function(
            allreduce_plugin,
            (to,),
            plugin_inputs.model_dump(),
        )
    to.replace_all_uses_with(allreduce, delete_user_cb=lambda user: user is not allreduce)


def get_shard_dim_idx(reshape: Reshape | Expand) -> int:
    """Get the shard dimension index of the reshape or expand node.

    Args:
        reshape (Reshape | Expand): The reshape or expand node

    Returns:
        int: The shard dimension index
    """
    dynamic_dim_idx = reshape.shape.index(-1)
    if dynamic_dim_idx == 0 and len(reshape.shape) == 2:
        shard_dim_idx = 1
    elif dynamic_dim_idx == 1 and len(reshape.shape) in [3, 4]:
        shard_dim_idx = 2
    elif dynamic_dim_idx == 2 and len(reshape.shape) in [3, 4]:
        shard_dim_idx = 1 if len(reshape.shape) == 4 else 0
    elif dynamic_dim_idx in [2, 3] and len(reshape.shape) == 5:
        shard_dim_idx = 1
    else:
        raise ValueError(f"unexpected shape of reformatting node{reshape.node.name}: {reshape.shape}")

    return shard_dim_idx
