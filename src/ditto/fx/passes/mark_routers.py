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

from torch.fx import Node

from ...configs import TRTLLMMapping
from ..subgraphs import MoESubgraph
from .infra import NodewiseOptimizationPass, NodewisePassResult


class MarkRouters(NodewiseOptimizationPass):
    """Mark router nodes in Mixture of Experts (MoE) layers.

    This pass identifies and marks router nodes in MoE layers to exclude them from tensor parallelism.

    It is a preprocessing of parallelize_linear pass required for proper conversion of MoE plugins
    with Tensor Parallelism.

    Attributes:
        mapping (TRTLLMMapping): The tensor parallel mapping configuration
    """

    mapping: TRTLLMMapping

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Mark router nodes in MoE layers.

        Args:
            node (Node): The node to identify MoE subgraph from

        Returns:
            dict[Node, NodewisePassResult]: Empty dict since no nodes are rewritten
        """
        if self.mapping.tp_size == 1 or not (moe := MoESubgraph.configure_from(node)):
            return {}

        moe.router.mark_expert_type_as("router")
        return {}
