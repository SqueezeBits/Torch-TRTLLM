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

from ..nodes import Cat, Stack, Unsqueeze
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateNopCatOrStack(NodewiseOptimizationPass):
    """Eliminate cat or stack called with just one input tensor."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (cat := Cat.specialize_from(node)) and len(cat.tensors) == 1:
            return {node: ReplaceAllUses(by=cat.tensors[0])}

        if (stack := Stack.specialize_from(node)) and len(stack.tensors) == 1:
            x = stack.tensors[0]
            graph = x.graph
            with graph.inserting_after(x):
                unsqueeze = Unsqueeze.create(graph, x, stack.dim)
            return {node: ReplaceAllUses(by=unsqueeze.node, propagate_meta=True)}

        return {}
