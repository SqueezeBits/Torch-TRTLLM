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

from ..nodes import Permute, Reshape, ToCopy
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class DeferCast(NodewiseOptimizationPass):
    """Defer casting operations after reshapes/permutes.

    This pass looks for patterns where a cast operation (ToCopy) is immediately followed by
    a reshape or permute operation. It reorders these operations to perform the reshape/permute
    first, followed by the cast. This is more efficient for simplifying pattern matching algorithms
    like Lora.

    For example, it transforms:
        y = torch.ops.aten._to_copy.default(x, ...)
        z = torch.ops.aten.reshape.default(y, ...)

    Into:
        y = torch.ops.aten.reshape.default(x, ...)
        z = torch.ops.aten._to_copy.default(y, ...)
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (to_copy := ToCopy.specialize_from(node))
            and len(users := list(to_copy.users)) == 1
            and (shuffle := Reshape.specialize_from(users[0]) or Permute.specialize_from(users[0]))
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            args_, kwargs_ = shuffle.args_kwargs(this=to_copy.this)
            pre_shuffle = type(shuffle).create(graph, *args_, **kwargs_)
            args_, kwargs_ = to_copy.args_kwargs(this=pre_shuffle)
            post_to_copy = ToCopy.create(graph, *args_, **kwargs_)
        return {shuffle.node: ReplaceAllUses(by=post_to_copy.node)}
