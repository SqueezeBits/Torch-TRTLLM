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

import torch
from torch.fx import Node

from ..nodes import MM, ToCopy
from ..subgraphs import MoESubgraph
from ..utils import forget_all_descendant_fake_tensors, get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class CastRouterToFP32(NodewiseOptimizationPass):
    """Pass that casts MoE router computations to FP32 for improved numerical stability.

    This pass identifies MoE router matrix multiplications and softmax operations that
    are not already in FP32 and inserts casting operations to perform the computation in FP32
    before casting back to the original dtype. This is the default behavior of TensorRT-LLM for MoE models.
    """

    @property
    def reversed_traversal(self) -> bool:
        """Whether to traverse nodes in reverse order."""
        return True

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        """Rewrite a node to cast MoE router computations to FP32.

        Args:
            node (Node): The node to potentially rewrite

        Returns:
            dict[Node, NodewisePassResult]: Mapping of nodes to their rewrite results.
                Empty if no rewrite was performed.
        """
        if not (
            (moe := MoESubgraph.configure_from(node))
            and (output_dtype := moe.router.mm.output_dtype)
            and output_dtype != torch.float32
            and (softmax_metadata := get_tensor_metadata(moe.softmax))
        ):
            return {}

        graph = node.graph
        with graph.inserting_before(moe.router.mm.node):
            lhs_cast = ToCopy.create(graph, moe.router.mm.this, dtype=torch.float32)
            rhs_cast = ToCopy.create(graph, moe.router.mm.other, dtype=torch.float32)
            router_fp32 = MM.create(graph, lhs_cast, rhs_cast)
            propagate_metadata_from(moe.router.mm, to=router_fp32)

            if softmax_metadata.dtype == torch.float32:
                output = ToCopy.create(graph, router_fp32, dtype=moe.router.mm.output_dtype).node
            else:
                output = router_fp32.node
                forget_all_descendant_fake_tensors(moe.router.mm.node)

        return {moe.router.mm.node: ReplaceAllUses(by=output, require_fake_tensor_prop=True)}
