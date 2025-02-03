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

from ..nodes import Reshape, View
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ReplaceViewByReshape(NodewiseOptimizationPass):
    """A replacement for the `view_to_reshape` pass in TorchTRT for its shape inference error."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (view := View.specialize_from(node)):
            return {}
        with (graph := node.graph).inserting_after(node):
            reshape = Reshape.create(graph, view.this, view.size)
            propagate_metadata_from(view, to=reshape)
        return {view.node: ReplaceAllUses(by=reshape.node)}
