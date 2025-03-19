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
from ..targets import AllReducePlugin
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses
from .parallelize_linear import insert_allreduce_plugin


class CanonicalizeMoEAllReduces(NodewiseOptimizationPass):
    """Canonicalize AllReduce operations in Mixture of Experts (MoE) layers.

    This pass consolidates AllReduce operations that follows down projection in each expert
    into a single AllReduce at the end of the MoE computation.

    It is a postprocessing of parallelize_linear pass required for proper conversion of MoE plugins
    with Tensor Parallelism.

    Attributes:
        mapping (TRTLLMMapping): The tensor parallel mapping configuration
    """

    mapping: TRTLLMMapping

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Rewrite MoE AllReduce operations to a canonical form.

        Args:
            node (Node): The node to potentially rewrite

        Returns:
            dict[Node, NodewisePassResult]: Mapping of nodes to their rewrite results

        Raises:
            NotImplementedError: If tensor parallelism is enabled but required AllReduce not found
        """
        if self.mapping.tp_size == 1 or not (moe := MoESubgraph.configure_from(node)):
            return {}

        graph = node.graph
        results: dict[Node, NodewisePassResult] = {}
        for expert in moe.experts:
            if not (
                (all_reduce := list(expert.down_proj.output_node.users)[0])
                and isinstance(all_reduce.target, AllReducePlugin)
            ):
                raise NotImplementedError(f"Tensor Parallelism is enabled but AllReduce not found for expert {expert}")
            results.update({all_reduce: ReplaceAllUses(by=all_reduce.all_input_nodes[0])})

        insert_allreduce_plugin(
            graph,
            moe.final_hidden_states,
            self.mapping.tp_group,
            strategy=all_reduce.target.strategy,  # type: ignore[union-attr]
            config=all_reduce.target.config,  # type: ignore[union-attr]
            fusion_op=all_reduce.target.fusion_op,  # type: ignore[union-attr]
            eps=all_reduce.target.eps,  # type: ignore[union-attr]
        )

        return results
