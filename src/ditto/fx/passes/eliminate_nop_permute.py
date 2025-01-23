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
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateNopPermute(NodewiseOptimizationPass):
    """Eliminate permute whose axis permutation is trivial."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((permute := Permute.specialize_from(node)) and permute.dims == [*range(permute.ndim)]):
            return {}
        return {node: ReplaceAllUses(by=permute.this)}
