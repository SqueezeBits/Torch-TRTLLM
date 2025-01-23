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

from ..nodes import SqueezeDim, Unsqueeze
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateUnsqueezeSqueeze(NodewiseOptimizationPass):
    """Eliminate unsqueeze followed by a squeeze with the same dim."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (squeeze := SqueezeDim.specialize_from(node))
            and (unsqueeze := Unsqueeze.specialize_from(squeeze.this))
            and (squeeze_dim := squeeze.nonnegative_dim) is not None
            and (unsqueeze_dim := unsqueeze.nonnegative_dim) is not None
            and squeeze_dim == unsqueeze_dim
        ):
            return {}
        return {node: ReplaceAllUses(by=unsqueeze.this)}
