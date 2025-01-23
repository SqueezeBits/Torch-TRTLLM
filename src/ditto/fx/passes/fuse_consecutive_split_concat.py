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

import operator

from torch.fx import Node

from ..nodes import Cat, Split
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseConsecutiveSplitConcat(NodewiseOptimizationPass):
    """Fuse consecutive split and concat that is identical to nop."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (split := Split.specialize_from(node))
            and (
                getitems := [
                    user for user in node.users if (user.op == "call_function" and user.target is operator.getitem)
                ]
            )
            and (
                (isinstance(split.split_size, int) and len(getitems) == split.split_size)
                or (isinstance(split.split_size, list) and len(getitems) == len(split.split_size))
            )
        ):
            return {}
        cat_nodes = {
            cat.node
            for getitem in getitems
            for user in getitem.users
            if (cat := Cat.specialize_from(user)) and cat.tensors == getitems and cat.dim == split.dim
        }
        return {cat_node: ReplaceAllUses(by=split.this) for cat_node in cat_nodes}
