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

from ..nodes import Clone, ToCopy
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class CanonicalizeCopy(NodewiseOptimizationPass):
    """Eliminate or simplify copy-like ops."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if clone := Clone.specialize_from(node):
            return {node: ReplaceAllUses(by=clone.this)}

        if copy := ToCopy.specialize_from(node):
            if copy.dtype_unchanged:
                return {node: ReplaceAllUses(by=copy.this)}

            if len(node.kwargs) > 1 and "dtype" in node.kwargs:
                node.kwargs = {"dtype": node.kwargs["dtype"]}
                return {node: ModifiedInsideThePass()}

        return {}
