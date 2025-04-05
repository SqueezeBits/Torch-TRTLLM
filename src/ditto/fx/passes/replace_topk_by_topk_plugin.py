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
from torch.fx import Node

from ...types import DataType
from ..nodes import TopK
from ..targets import MixtureOfExpertsPlugin, TopkLastDimPlugin
from ..utils import find_nearest, get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class ReplaceTopkByTopkLastDimPlugin(NodewiseOptimizationPass):
    """Pass that replaces torch.topk operations with TopkLastDimPlugin.

    This pass identifies topk operations that can be replaced with TopkLastDimPlugin.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Rewrite topk operations to use TopkLastDimPlugin.

        Args:
            node (Node): The node to potentially rewrite

        Returns:
            dict[Node, NodewisePassResult]: Mapping of nodes to their rewrite results

        Raises:
            NotImplementedError: If input tensor has dimensions other than 2
        """
        # NOTE: Currently, this pass only converts topk operations in group_limited_greedy method of MoE,
        #       because TopkLastDimPlugin in trtllm is not an always-enabled plugin, which is enabled
        #       by prefer_plugin flag. It's True for the topk operations in group_limited_greedy.
        if not (
            isinstance(node.target, MixtureOfExpertsPlugin)
            and (group_topk := find_nearest(TopK, node, follow_first_only=False))
            and (input_ndim := group_topk.input_ndim)
            and (group_topk.dim in (-1, input_ndim - 1))
            and (input_metadata := get_tensor_metadata(group_topk.this))
        ):
            return {}

        if input_ndim != 2:
            raise NotImplementedError(
                f"TopkLastDimPlugin conversion with input_ndim {input_ndim} != 2 is not implemented yet."
            )

        graph = node.graph
        topk_last_dim_plugin = TopkLastDimPlugin(
            k=int(group_topk.k),
            type_id=DataType(input_metadata.dtype).to(trt.DataType),
        )
        with graph.inserting_after(group_topk.this):
            plugin_node = graph.call_function(
                topk_last_dim_plugin,
                (group_topk.this,),
            )

        return {group_topk.node: ReplaceAllUses(by=plugin_node)}
