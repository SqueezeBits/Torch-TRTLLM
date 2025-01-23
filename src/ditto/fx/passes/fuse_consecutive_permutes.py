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

from ..nodes import Permute
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class FuseConsecutivePermutes(NodewiseOptimizationPass):
    """Fuse two consecutive permutes."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (permute := Permute.specialize_from(node))
            and (
                children := [
                    child_permute
                    for user in node.users
                    if ((child_permute := Permute.specialize_from(user)) and child_permute.ndim == permute.ndim)
                ]
            )
        ):
            return {}
        results: dict[Node, NodewisePassResult] = {}
        graph = node.graph
        for child_permute in children:
            # e.g. (N, C, H, W)  -[0, 3, 1, 2]-> [N, W, C, H] -[0, 2, 1, 3]-> (N, C, W, H)
            # is equivalent to (N, C, H, W) -[0, 1, 3, 2]-> (N, C, W, H)
            dims = [permute.dims[child_permute.dims[i]] for i in range(permute.ndim)]
            with graph.inserting_after(child_node := child_permute.node):
                fused_permute = Permute.create(graph, permute.this, dims)
                inject_stack_trace_from(child_permute, permute, to=fused_permute)
            results[child_node] = ReplaceAllUses(by=fused_permute.node)
        return results
