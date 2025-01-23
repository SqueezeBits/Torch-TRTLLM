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

from ..nodes import MulTensor, PowTensorScalar
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class RewritePowAsMul(NodewiseOptimizationPass):
    """Rewrite pow op as mul op with self.

    Required to prevent engine build failures due to casts inserted where computations are in bf but literals remain fp.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (power := PowTensorScalar.specialize_from(node)) and power.other == 2:
            graph = node.graph

            with graph.inserting_before(node):
                equivalent_mul = MulTensor.create(graph, power.this, power.this)
                inject_stack_trace_from(node, to=equivalent_mul.node)

            return {node: ReplaceAllUses(by=equivalent_mul.node)}
        return {}
