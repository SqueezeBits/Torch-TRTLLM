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
# mypy: disable-error-code="union-attr"

from loguru import logger
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from torch.fx import Node

from ...configs import TRTLLMMapping
from ...types import expect_identical
from ..nodes import AddTensorTensor
from ..subgraphs import MoESubgraph
from ..targets import AllReducePlugin
from ..utils import find_nearest
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses
from .parallelize_linear import insert_allreduce_plugin


class CanonicalizeMoEAllReduces(NodewiseOptimizationPass):
    """Canonicalize AllReduce operations in Mixture of Experts (MoE) layers.

    This pass consolidates AllReduce operations that follows down projection in each expert
    into a single AllReduce at the end of the MoE computation. Furthermore, if there is a shared expert,
    it also consolidates the AllReduce at the end of the shared expert into the consolidated AllReduce.

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
        all_reduce_input = moe.final_hidden_states
        for expert in moe.experts:
            if not (
                (all_reduce := list(expert.down_proj.output_node.users)[0])
                and isinstance(all_reduce.target, AllReducePlugin)
            ):
                raise NotImplementedError(f"Tensor Parallelism is enabled but AllReduce not found for expert {expert}")
            results.update({all_reduce: ReplaceAllUses(by=all_reduce.all_input_nodes[0])})

        if (
            moe.shared_experts
            and (down_proj := moe.shared_experts[0][2])
            and (shared_expert_all_reduce := list(down_proj.output_node.users)[0])
            and isinstance(shared_expert_all_reduce.target, AllReducePlugin)
            and (
                add := find_nearest(
                    AddTensorTensor, shared_expert_all_reduce, follow_parent=False, follow_first_only=False
                )
            )
        ):
            results.update({shared_expert_all_reduce: ReplaceAllUses(by=shared_expert_all_reduce.all_input_nodes[0])})
            all_reduce_input = add.node
            if not (
                expect_identical(
                    all_reduce.target.strategy,
                    shared_expert_all_reduce.target.strategy,
                    expecting_type=AllReduceStrategy,
                )
                is not None
                and expect_identical(
                    all_reduce.target.fusion_op,
                    shared_expert_all_reduce.target.fusion_op,
                    expecting_type=AllReduceFusionOp,
                )
                is not None
                and expect_identical(all_reduce.target.eps, shared_expert_all_reduce.target.eps, expecting_type=float)
                is not None
            ):
                logger.warning(
                    f"AllReduce configurations are different for expert and shared expert. "
                    f"Using expert's configurations for consolidated AllReduce."
                    f"\n\tExpert: {all_reduce.target.strategy}, "
                    f"{all_reduce.target.fusion_op}, {all_reduce.target.eps}"
                    f"\n\tShared expert: {shared_expert_all_reduce.target.strategy}, "
                    f"{shared_expert_all_reduce.target.fusion_op}, {shared_expert_all_reduce.target.eps}"
                )

        insert_allreduce_plugin(
            graph,
            all_reduce_input,
            self.mapping.tp_group,
            strategy=all_reduce.target.strategy,
            fusion_op=all_reduce.target.fusion_op,
            eps=all_reduce.target.eps,
        )

        return results
