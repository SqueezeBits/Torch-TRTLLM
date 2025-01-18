from copy import copy

import tensorrt as trt
import torch
from tensorrt_llm.functional import AllReduceConfig, AllReduceFusionOp, AllReduceStrategy
from torch.fx import Graph, GraphModule, Node
from typing_extensions import Self

from ...configs.trtllm.pretrained import TRTLLMMapping
from ...types import DataType, StrictlyTyped
from ..nodes import GetAttr, Permute, Reshape
from ..subgraphs.linear import Linear, find_last_linear, find_nearest_linear_projection
from ..targets import AllGatherPlugin, AllReducePlugin, AllReducePluginInputs, GPTAttentionPlugin
from ..utils import (
    find_closest_common_descendant,
    forget_all_descendant_fake_tensors,
    get_descendants_with_depth,
    get_val,
    replace_all_uses_with,
)
from .infra import (
    GraphOptimizationPass,
    PassResult,
    inject_stack_trace_from,
)


# TODO: Change ParallelizeTensor to inherit from NodewiseOptimization instead of GraphOptimizationPass
# This will allow processing nodes individually rather than the whole graph at once
class ParallelizeTensor(GraphOptimizationPass):
    """Parallelize tensor in the graph (Tensor Parallelism).

    Attributes:
        mapping (TRTLLMMapping): The mapping of the model
        first_node_rewritten (Node | None): The first node that is rewritten
    """

    mapping: TRTLLMMapping
    first_node_rewritten: Node | None = None

    def call(self, graph_module: GraphModule) -> PassResult:
        overall_modified = False
        for node in graph_module.graph.nodes:
            if not (pattern := DecoderPattern.extract_from(node)):
                continue
            assert (
                pattern.plugin.num_heads % self.mapping.tp_size == 0
            ), "num_attention_heads must be divisible by tp_size"

            if self.first_node_rewritten is None:
                self.first_node_rewritten = pattern.attn_qkv.input_node

            hidden_size = pattern.plugin.head_size * pattern.plugin.num_heads
            attention_head_size = hidden_size // pattern.plugin.num_heads
            num_attention_heads = pattern.plugin.num_heads // self.mapping.tp_size
            num_attention_kv_heads = (pattern.plugin.num_kv_heads + self.mapping.tp_size - 1) // self.mapping.tp_size

            # update plugin's fields and val's shape
            pattern.plugin.num_heads = num_attention_heads
            pattern.plugin.num_kv_heads = num_attention_kv_heads
            pattern.plugin.tp_size = self.mapping.tp_size
            pattern.plugin.tp_rank = self.mapping.tp_rank

            # parallelize the qkv linear
            in_features = hidden_size
            q_dim = self.mapping.tp_size * num_attention_heads * attention_head_size
            kv_dim = self.mapping.tp_size * num_attention_kv_heads * attention_head_size
            out_features = q_dim + (2 * kv_dim)
            self.parallelize_column_linear(
                graph_module.graph,
                pattern.attn_qkv,
                in_features,
                out_features,
                gather_output=False,
                is_qkv=True,
                q_dim=q_dim,
                kv_dim=kv_dim,
            )

            # parallelize the dense linear
            in_features = self.mapping.tp_size * num_attention_heads * attention_head_size
            out_features = hidden_size
            self.parallelize_row_linear(
                graph_module.graph,
                pattern.attn_dense,
                in_features,
                out_features,
            )

            # parallelize the mlp gate linear
            ffn_hidden_size = get_val(pattern.mlp_gate.output_node).shape[-1]
            in_features = hidden_size
            out_features = ffn_hidden_size
            self.parallelize_column_linear(
                graph_module.graph,
                pattern.mlp_gate,
                in_features,
                out_features,
                gather_output=False,
            )

            # parallelize the mlp up-projection linear
            in_features = hidden_size
            out_features = (
                ffn_hidden_size  # Note: it should be multiplied by 2 if hidden activation is 'swiglu' or 'gegelu'
            )
            self.parallelize_column_linear(
                graph_module.graph,
                pattern.mlp_up_proj,
                in_features,
                out_features,
                gather_output=False,
            )

            # parallelize the mlp down-projection linear
            in_features = ffn_hidden_size
            out_features = hidden_size
            self.parallelize_row_linear(
                graph_module.graph,
                pattern.mlp_down_proj,
                in_features,
                out_features,
            )

            overall_modified = overall_modified or True

        # parallelize the lm_head linear
        if (
            (lm_head := find_last_linear(graph_module.graph))
            and (lm_head_weight := GetAttr.specialize_from(lm_head.weight_node))
            and (len(lm_head_weight.tensor.shape) == 2)
        ):
            in_features = (
                lm_head_weight.tensor.shape[1] if lm_head.has_transposed_weight else lm_head_weight.tensor.shape[0]
            )
            out_features = (
                lm_head_weight.tensor.shape[0] if lm_head.has_transposed_weight else lm_head_weight.tensor.shape[1]
            )
            self.parallelize_column_linear(
                graph_module.graph,
                lm_head,
                in_features,
                out_features,
                gather_output=True,
            )

        forget_all_descendant_fake_tensors(self.first_node_rewritten)

        return PassResult(graph_module=graph_module, modified=overall_modified, require_fake_tensor_prop=True)

    def get_name_of_attr(self, from_name: str) -> str:
        """Create the name of the attribute with the tensor parallel rank.

        Args:
            from_name (str): The name of the attribute

        Returns:
            str: The name of the attribute with the tensor parallel rank
        """
        return f"{from_name}_rank{self.mapping.tp_rank}"

    # pylint: disable-next=too-many-locals
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
            gather_output (bool): Whether to gather the output
            is_qkv (bool): Whether the linear subgraph is a QKV linear
            q_dim (int): The dimension of the query
            kv_dim (int): The dimension of the key/value
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
        ), "unexpected shape of parallelized qkv weight"

        with graph.inserting_before(weight.node):
            parallelized_weight = GetAttr.create(
                graph, self.get_name_of_attr(weight.target), parallelized_weight_tensor
            )
            inject_stack_trace_from(weight, to=parallelized_weight)
        weight.node.replace_all_uses_with(parallelized_weight.node)
        linear.mm.other = parallelized_weight.node

        if linear.bias_node is not None:
            bias = GetAttr.specialize_from(linear.bias_node)
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
            strategy (AllReduceStrategy): The strategy of the allreduce plugin
            config (AllReduceConfig): The config of the allreduce plugin
            fusion_op (AllReduceFusionOp): The fusion operation of the allreduce plugin
            eps (float): The epsilon value of the allreduce plugin
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
        ), "unexpected shape of parallelized qkv weight"

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
        gather_dim (int): The dimension to gather
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

    replace_all_uses_with(to, reshape_2.node, exclude_nodes=[allgather])


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
        strategy (AllReduceStrategy): The strategy of the allreduce plugin
        config (AllReduceConfig): The config of the allreduce plugin
        fusion_op (AllReduceFusionOp): The fusion operation of the allreduce plugin
        eps (float): The epsilon value of the allreduce plugin

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
    replace_all_uses_with(to, allreduce)


class DecoderPattern(StrictlyTyped):
    """A pattern representing a decoder layer of GPT model.

    This pattern is used to extract the linear subgraphs based on the GPT attention plugin node.
    1) Find the GPTAttentionPlugin node
    2) Find the nearest linear subgraph to the GPTAttentionPlugin node
       by using find_nearest_linear_projection (QKV linear)
    3) Find the descendant linear subgraphs by using find_descendant_linears
       (Dense linear, MLP gate, MLP up-projection, MLP down-projection)

    Attributes:
        plugin (GPTAttentionPlugin): The GPT attention plugin node
        attn_qkv (Linear): The QKV linear layer
        attn_dense (Linear): The dense linear layer
        mlp_gate (Linear): The MLP gate linear layer
        mlp_up_proj (Linear): The MLP up-projection linear layer
        mlp_down_proj (Linear): The MLP down-projection linear layer
    """

    plugin: GPTAttentionPlugin
    attn_qkv: Linear
    attn_dense: Linear
    mlp_gate: Linear
    mlp_up_proj: Linear  # Note: mlp_gate and mlp_up_proj may be fused in the future
    mlp_down_proj: Linear

    @classmethod
    def extract_from(cls, node: Node) -> Self | None:
        """Extract configuration from a GPT attention plugin node.

        Recursively analyzes the plugin node's ancestors and predecessors to find the associated
        linear/gemm operations for attention and feed forward network.

        Args:
            node: Node expected to be a GPTAttentionPlugin

        Returns:
            DecoderPattern if successfully extracted, None otherwise
        """
        if not isinstance(node.target, GPTAttentionPlugin):
            return None
        # Note: We need to copy the plugin node's target to avoid modifying the original.
        # This is because GraphModule's deepcopy method creates new nodes rather than copying existing ones.
        node.target = plugin = copy(node.target)
        attn_qkv = find_nearest_linear_projection(node)
        if (found_linears := find_descendant_linears(node)) is None:
            raise RuntimeError("Failed to find all linear subgraphs associated with GPTAttentionPlugin")
        attn_dense, maybe_mlp_gate, maybe_mlp_up_proj, mlp_down_proj = found_linears
        assert (
            attn_qkv and attn_dense and maybe_mlp_gate and maybe_mlp_up_proj and mlp_down_proj
        ), "Failed to find all linear subgraphs"

        mlp_common_descendant = find_closest_common_descendant(maybe_mlp_gate.input_node, maybe_mlp_up_proj.input_node)
        assert mlp_common_descendant, "Failed to find common descendant"

        expected_common_descendant = list(maybe_mlp_gate.output_node.users)[0]
        if expected_common_descendant == mlp_common_descendant:
            mlp_gate, mlp_up_proj = maybe_mlp_gate, maybe_mlp_up_proj
        else:
            mlp_gate, mlp_up_proj = maybe_mlp_up_proj, maybe_mlp_gate

        return cls(
            plugin=plugin,
            attn_qkv=attn_qkv,
            attn_dense=attn_dense,
            mlp_gate=mlp_gate,
            mlp_up_proj=mlp_up_proj,
            mlp_down_proj=mlp_down_proj,
        )


def find_descendant_linears(node: Node) -> tuple[Linear, Linear, Linear, Linear] | None:
    """Find the descendant linear subgraphs from GPT attention plugin node.

    The layers found are expected to be a dense linear of attention layer and
    3 layers(gate, up-projection, down-projection) of feed forward network layer.

    Args:
        node: The node expected to be a GPTAttentionPlugin node for searching the descendant linear subgraphs

    Returns:
        The descendant linear subgraphs
    """
    if (
        not (
            descendant_linear_subgraphs := {
                subgraph: depth
                for node, depth in get_descendants_with_depth(node).items()
                if (subgraph := Linear.configure_from(node))
            }
        )
        and len(descendant_linear_subgraphs) < 4
    ):
        return None
    sorted_subgraphs = sorted(descendant_linear_subgraphs, key=lambda subgraph: descendant_linear_subgraphs[subgraph])
    return tuple(sorted_subgraphs[0:4])
