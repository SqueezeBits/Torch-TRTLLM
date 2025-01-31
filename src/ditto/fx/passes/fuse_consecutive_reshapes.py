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

from ..nodes import Reshape, SingleDimensionReshape
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAmongInputs


class FuseConsecutiveReshapes(NodewiseOptimizationPass):
    """Fuse two consecutive reshapes."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (parent := Reshape.specialize_from(node) or SingleDimensionReshape.specialize_from(node))
            and (
                children := [
                    child_reshape for child_node in node.users if (child_reshape := Reshape.specialize_from(child_node))
                ]
            )
        ):
            return {}
        return {child.node: ReplaceAmongInputs(occurrences_of=node, by=parent.this) for child in children}
