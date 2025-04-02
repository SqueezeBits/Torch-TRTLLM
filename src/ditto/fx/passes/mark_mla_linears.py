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
from ..nodes import ScaledDotProductAttention
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult
from .replace_sdpa_by_gpt_attention_plugin import MLA


class MarkMLALinears(NodewiseOptimizationPass):
    """Mark linear nodes in MLA layers with the corresponding type to exclude them from tensor parallelism.

    It is a preprocessing of parallelize_linear pass required for proper conversion of MLA subgraphs
    with Tensor Parallelism.

    Attributes:
        mapping (TRTLLMMapping): The tensor parallel mapping configuration
    """

    mapping: TRTLLMMapping

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Mark linear nodes in MLA layers with the corresponding type.

        Args:
            node (Node): The node to identify MLA subgraph from

        Returns:
            dict[Node, NodewisePassResult]: Empty dict since no nodes are rewritten
        """
        if not (
            self.mapping.tp_size > 1
            and (sdpa := ScaledDotProductAttention.specialize_from(node))
            and (mla := MLA.extract_from(sdpa))
        ):
            return {}
        mla.kv_a_proj.mark_linear_type_as("mla_kv_a_proj")
        mla.kv_b_proj.mark_linear_type_as("mla_kv_b_proj")
        mla.q_proj.mark_linear_type_as("mla_q_proj")
        mla.o_proj.mark_linear_type_as("mla_o_proj")
        return {node: ModifiedInsideThePass()}
