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
from ..nodes import MulTensorTensor
from ..subgraphs import Linear, MoESubgraph
from ..utils import find_nearest
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult


class MarkMoELinears(NodewiseOptimizationPass):
    """Mark linear nodes in MoE layers with the corresponding type to exclude them from tensor parallelism.

    It is a preprocessing of parallelize_linear pass required for proper conversion of MoE plugins
    with Tensor Parallelism.

    Attributes:
        mapping (TRTLLMMapping): The tensor parallel mapping configuration
    """

    mapping: TRTLLMMapping

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Mark linear nodes in MoE layers with the corresponding type.

        Args:
            node (Node): The node to identify MoE subgraph from

        Returns:
            dict[Node, NodewisePassResult]: Empty dict since no nodes are rewritten
        """
        if self.mapping.tp_size == 1 or not (moe := MoESubgraph.configure_from(node)):
            return {}

        moe.router.mark_linear_type_as("router")
        for _, _, down_proj in moe.shared_experts:
            if shared_expert_gating := MulTensorTensor.specialize_from(list(down_proj.output_node.users)[0]):
                # Normally, the output of shared expert is directly added to the output of MoE layers.
                # In some models(e.g. Qwen) however, there is an additional shared expert gate.
                if shared_expert_gating.this == down_proj.mm.node:
                    shared_expert_gate = find_nearest(Linear, shared_expert_gating.other)
                else:
                    shared_expert_gate = find_nearest(Linear, shared_expert_gating.this)
                if shared_expert_gate is None:
                    raise NotImplementedError(f"Unsupported shared expert gate found from: {shared_expert_gating}")
                shared_expert_gate.mark_linear_type_as("shared_expert_gate")

        return {node: ModifiedInsideThePass()}
