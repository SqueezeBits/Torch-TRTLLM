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

from torch.fx.node import Node

from ..nodes import GetItem, Slice, SplitTensor
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class RewriteSplitAsSlices(NodewiseOptimizationPass):
    """Rewrite a split node as a group of slices."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (split := SplitTensor.specialize_from(node))
            and isinstance(size := split.split_size, int)
            and (getitems := [getitem for user in node.users if (getitem := GetItem.specialize_from(user))])
            and len(getitems) == len(node.users)
        ):
            return {}

        graph = node.graph
        results: dict[Node, NodewisePassResult] = {}
        with graph.inserting_before(node):
            for getitem in getitems:
                s = Slice.create(
                    graph,
                    split.this,
                    split.dim,
                    getitem.idx * size,
                    (getitem.idx + 1) * size,
                )
                propagate_metadata_from(node, to=s)
                results[getitem.node] = ReplaceAllUses(by=s.node)
        return results
