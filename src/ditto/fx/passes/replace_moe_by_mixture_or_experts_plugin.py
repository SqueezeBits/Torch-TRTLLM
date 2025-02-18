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
from loguru import logger
from torch.fx import Graph, Node
from transformers import PretrainedConfig

from ...types import DataType, verify
from ..subgraphs import MoESubgraph
from ..targets import MixtureOfExpertsPlugin, MixtureOfExpertsPluginInputs, MoEConfig
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class ReplaceMoEByMoEPlugin(NodewiseOptimizationPass):

    dtype: torch.dtype
    has_warned_missing_pretrained_config: bool = False

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (moe := MoESubgraph.configure_from(node)):
            return {}

        graph = node.graph

        pretrained_config: PretrainedConfig | None = (
            verify(
                graph_module.meta.get("pretrained_config"),
                as_type=PretrainedConfig,
            )
            if (graph_module := graph.owning_module)
            else None
        )

        if not self.has_warned_missing_pretrained_config and pretrained_config is None:
            logger.warning("No pretrained config found in graph module meta data. Default MoE config will be used.")
            self.has_warned_missing_pretrained_config = True

        moe_config = MoEConfig.from_pretrained_config(pretrained_config)
        moe_plugin = MixtureOfExpertsPlugin(
            **moe_config.model_dump(),
            activation_type=3, # TODO: remove hard-coded activation type
            type_id=DataType(self.dtype).to(trt.DataType),
            weight_type_id=DataType(self.dtype).to(trt.DataType),
            output_type_id=DataType(self.dtype).to(trt.DataType),
        )
        with graph.inserting_before(moe.final_hidden_states):
            plugin_inputs = MixtureOfExpertsPluginInputs.create_from(moe, graph)
            plugin_node = graph.call_function(
                moe_plugin,
                (),
                plugin_inputs.model_dump(),
            )
        self.remove_unused_nodes(moe, graph)

        return {moe.final_hidden_states: ReplaceAllUses(by=plugin_node)}

    def remove_unused_nodes(self, moe: MoESubgraph, graph: Graph):
        for node in moe.unused_nodes:
            graph.erase_node(node)
