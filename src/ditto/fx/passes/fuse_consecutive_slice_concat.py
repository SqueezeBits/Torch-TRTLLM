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

from ..nodes import Cat, Slice
from ..nodes.aten.utils import has_same_values
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseConsecutiveSliceConcat(NodewiseOptimizationPass):
    """Fuse consecutive slices and concat that is identical to nop."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (cat := Cat.specialize_from(node))
            and (slices := Slice.sort([s for x in cat.tensors if (s := Slice.specialize_from(x))]))
            and len(slices) == len(cat.tensors)
            and has_same_values(slices[0].nonnegative_dim, cat.nonnegative_dim)
            and Slice.are_consecutive(slices)
        ):
            return {}
        return {cat.node: ReplaceAllUses(by=slices[0].this)}
