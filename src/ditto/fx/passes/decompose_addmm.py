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

from ..nodes import MM, AddMM, AddTensor
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class DecomposeAddMM(NodewiseOptimizationPass):
    """Decompose addmm into mm and add."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (addmm := AddMM.specialize_from(node)):
            return {}

        with (graph := node.graph).inserting_before(node):
            mm = MM.create(graph, addmm.mat1, addmm.mat2)
            propagate_metadata_from(addmm, to=mm)
            add = AddTensor.create(graph, mm.node, addmm.this)
            propagate_metadata_from(addmm, to=add)
        return {node: ReplaceAllUses(by=add.node)}
