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

from ..nodes import Index
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class RewriteIndexAsSingleSlice(NodewiseOptimizationPass):
    """Rewrite index op as single slice op when possible (required to support models with MQA)."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (index := Index.specialize_from(node)) and index.can_replace_with_single_slice:
            graph = node.graph

            dim = index.dim
            start = index.idx
            end = start + 1

            with graph.inserting_before(node):
                equivalent_slice = graph.call_function(torch.ops.aten.slice.Tensor, (index.this, dim, start, end))
                propagate_metadata_from(node, to=equivalent_slice)

            return {node: ReplaceAllUses(by=equivalent_slice)}
        return {}
