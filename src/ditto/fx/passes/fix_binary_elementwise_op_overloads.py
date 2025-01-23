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

from ..nodes.aten import BinaryElementwise
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FixBinaryElementwiseOpOverloads(NodewiseOptimizationPass):
    """Fix binary elementwise operator overloads that failed to specialize.

    This pass attempts to re-specialize binary elementwise operators that has been incorrectly specialized
    by explicitly creating them from their overload packet. This handles cases where the initial
    specialization may have failed due to type mismatches or other issues.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        with node.graph.inserting_before(node):
            if not (
                node.target in BinaryElementwise.possible_targets()
                and BinaryElementwise.specialize_from(node) is None
                and (
                    replacement := BinaryElementwise.create_from_overloadpacket(
                        node.graph,
                        args=node.args,
                        kwargs=node.kwargs,
                        overloadpacket=node.target.overloadpacket,
                    )
                )
            ):
                return {}
        inject_stack_trace_from(node, to=replacement)
        return {node: ReplaceAllUses(by=replacement.node)}
