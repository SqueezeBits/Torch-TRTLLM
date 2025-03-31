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

import torch
from torch.fx import Node

from ..nodes import GetAttr, MulTensorScalar
from ..utils import get_val
from .infra import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult


class OverrideMulScalarTypePromotion(NodewiseOptimizationPass):
    """Cast the scalar (second input node) of a mul node to the dtype of the fisrt input node.

    This pass is to suppress the type promotion of float16/bfloat16 tensors to float32 by torch-tensorrt converter.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (mul := MulTensorScalar.specialize_from(node))
            and isinstance((input_val := get_val(mul.this)), torch.Tensor)
            and (input_val.dtype in (torch.float16, torch.bfloat16))
            and isinstance(mul.other, float)
        ):
            return {}

        graph = node.graph
        with graph.inserting_before(node):
            rhs_cast = GetAttr.create(graph, f"{mul.name}_rhs_cast", torch.tensor(mul.other, dtype=input_val.dtype))
            args, kwargs = mul.args_kwargs(other=rhs_cast.node)
            mul.node.args = args
            mul.node.kwargs = kwargs

        return {node: ModifiedInsideThePass()}
