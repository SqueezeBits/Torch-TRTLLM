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
from torch.fx import Graph, GraphModule, Node

from ...types import DataType
from ..metadata_keys import MOE_CONFIG
from ..subgraphs import MoESubgraph
from ..targets import (
    MixtureOfExpertsPlugin,
    MixtureOfExpertsPluginInputs,
    get_moe_activation_type,
    get_moe_normalization_mode,
)
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, get_pretrained_config


class ReplaceMoEByMoEPlugin(NodewiseOptimizationPass):
    """Pass that replaces Mixture of Experts (MoE) subgraphs with TensorRT MoE plugin nodes.

    Attributes:
        dtype (torch.dtype): Data type to use for the plugin tensors
    """

    dtype: torch.dtype
    tp_size: int
    tp_rank: int
    plugin: MixtureOfExpertsPlugin | None = None

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Rewrite a node by replacing MoE subgraph with plugin node if applicable.

        Args:
            node (Node): Node to potentially rewrite

        Returns:
            dict[Node, NodewisePassResult]: Mapping from nodes to their rewrite results
        """
        if not (moe := MoESubgraph.configure_from(node)):
            return {}

        graph = node.graph
        pretrained_config = get_pretrained_config(graph)

        moe_plugin = MixtureOfExpertsPlugin(
            number_of_experts=moe.number_of_experts,
            top_k=moe.top_k,
            expert_hidden_size=moe.expert_hidden_size,
            expert_inter_size=moe.expert_inter_size,
            normalization_mode=get_moe_normalization_mode(pretrained_config),
            activation_type=get_moe_activation_type(),
            type_id=DataType(self.dtype).to(trt.DataType),
            weight_type_id=DataType(self.dtype).to(trt.DataType),
            output_type_id=DataType(self.dtype).to(trt.DataType),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        moe_plugin._shared_expert_intermediate_size = moe.shared_expert_intermediate_size
        if self.plugin is None:
            self.plugin = moe_plugin

        with graph.inserting_before(moe.final_hidden_states):
            plugin_inputs = MixtureOfExpertsPluginInputs.create_from(moe, graph)
            plugin_node = graph.call_function(
                moe_plugin,
                (),
                plugin_inputs.model_dump(),
            )
        self.remove_unused_nodes(moe, graph)

        return {moe.final_hidden_states: ReplaceAllUses(by=plugin_node)}

    def remove_unused_nodes(self, moe: MoESubgraph, graph: Graph) -> None:
        """Remove nodes from the graph that are no longer used after plugin replacement.

        Args:
            moe (MoESubgraph): MoE subgraph containing unused nodes
            graph (Graph): Graph to remove nodes from
        """
        # NOTE: The `torch.ops.aten.sym_constrain_range_for_size.default` nodes in experts are not used but
        # are not eliminated by PyTorch's DCE process since they are side effectful nodes. This causes their ancestor
        # nodes (including topk and nonzero, which don't have torch-tensorrt conversion logic yet) to also remain.
        # These nodes should be eliminated to match TensorRT-LLM's network structure and to enable successful
        # conversion. It has been verified that removing them is safe and does not affect functionality.
        for node in moe.unused_nodes:
            graph.erase_node(node)

    def postprocess(self, graph_module: GraphModule) -> None:
        super().postprocess(graph_module)
        if self.plugin is not None:
            graph_module.meta[MOE_CONFIG] = {
                "num_experts": self.plugin.number_of_experts,
                "shared_expert_intermediate_size": self.plugin._shared_expert_intermediate_size,
                "top_k": self.plugin.top_k,
                "normalization_mode": self.plugin.normalization_mode,
                "sparse_mixer_epsilon": self.plugin.sparse_mixer_epsilon,
            }
