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

from torch import Size, SymInt, Tensor
from torch.fx import Node

from ...types import ShapeArg
from ..nodes import Expand, Reshape, SymSizeInt
from ..utils import get_val
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult


class ResolveDynamicReshape(NodewiseOptimizationPass):
    """Resolve reshape operations containing a single dynamic(symbolic int) dimension.

    It replaces the dynamic dimension with the automatic inference value (-1) when the target reshape operation
    has a single symbolic dimension. If there is already a inference value, it will be replaced with the proper value.

    Example1:
        Before: reshape(x, [dim0, symint, dim2])
        After: reshape(x, [dim0, -1, dim2])

    Example2:
        Before: reshape(x, [dim0, symint, -1])
        After: reshape(x, [dim0, -1, dim2])
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if (
            (reshape := Reshape.specialize_from(node) or Expand.specialize_from(node))
            and (num_auto_infer_values := len([dim for dim in reshape.shape if dim == -1])) <= 1
            and len([dim for dim in reshape.shape if isinstance(dim, Node) and SymSizeInt.specialize_from(dim)]) == 1
        ):
            preprocessed_shape = reshape.shape[:]
            if (
                num_auto_infer_values == 1
                and (output_val := get_val(node, expected_type=Tensor)) is not None
                and is_resolvable(reshape.shape, output_val.shape)
            ):
                for i, output_dim in enumerate(output_val.shape):
                    if isinstance(preprocessed_shape[i], int) and preprocessed_shape[i] == -1:
                        preprocessed_shape[i] = output_dim

            new_shape = [dim if isinstance(dim, int) else -1 for dim in preprocessed_shape]
            args, kwargs = reshape.args_kwargs(shape=new_shape)
            reshape.node.args = args
            reshape.node.kwargs = kwargs
            return {node: ModifiedInsideThePass()}

        return {}


def is_resolvable(shape_arg: ShapeArg, output_shape: Size) -> bool:
    """Check if the shape can be resolved to the output shape.

    Args:
        shape_arg (ShapeArg): The shape argument provided to reshape node.
        output_shape (Size): The output shape to check against.
    """
    if len(shape_arg) != len(output_shape):
        return False

    for dim, output_dim in zip(shape_arg, output_shape):
        if isinstance(dim, int) and dim == -1:
            continue

        # pylint: disable-next=too-many-boolean-expressions
        if (isinstance(dim, int) and isinstance(output_dim, int) and dim != output_dim) or (
            (isinstance(dim, Node) and (symint_dim_val := get_val(dim)) and isinstance(symint_dim_val, SymInt))
            and (isinstance(output_dim, SymInt) and symint_dim_val != output_dim)
        ):
            return False

    return True
