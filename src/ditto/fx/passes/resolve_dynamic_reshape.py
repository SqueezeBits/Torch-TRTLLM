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

from ..nodes import Reshape, SymSizeInt
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult


class ResolveDynamicReshape(NodewiseOptimizationPass):
    """Resolve reshape operations containing a single dynamic(symbolic) dimension.

    It replaces the dynamic dimension with the automatic inference value (-1) when the target reshape operation
    has a single symbolic dimension.

    Example:
        Before: reshape(x, [dim0, dynamic_dim, dim2])
        After: reshape(x, [dim0, -1, dim2])
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (
            (reshape := Reshape.specialize_from(node))
            and -1 not in reshape.shape
            and (len([dim for dim in reshape.shape if isinstance(dim, Node) and SymSizeInt.specialize_from(dim)]) == 1)
        ):
            new_shape = [dim if isinstance(dim, int) else -1 for dim in reshape.shape]
            args, kwargs = reshape.args_kwargs(shape=new_shape)
            reshape.node.args = args
            reshape.node.kwargs = kwargs
            return {node: ModifiedInsideThePass()}
        return {}
