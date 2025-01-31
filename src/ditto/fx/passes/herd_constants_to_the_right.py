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

from ...types import Number
from ..nodes import Binary
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult


class HerdConstantsToTheRight(NodewiseOptimizationPass):
    """Herd constant inputs of binary nodes to the right hand side."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (binary := Binary.specialize_from(node))
            and binary.is_commutative
            and isinstance(binary.this, Number)
            and len(node.args) >= 2
        ):
            return {}
        node.args = (node.args[1], node.args[0], *node.args[2:])
        return {node: ModifiedInsideThePass()}
