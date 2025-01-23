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

from ..nodes import Reshape, Unsqueeze
from ..utils import get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class RewriteReshapeAsUnsqueeze(NodewiseOptimizationPass):
    """Rewrite reshape as unsqueeze if possible."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (reshape := Reshape.specialize_from(node))
            and (input_tensor := get_tensor_metadata(reshape.this))
            and (output_tensor := get_tensor_metadata(reshape.node)) is not None
            and (dim := find_unsqueeze_dim(input_tensor.shape, output_tensor.shape)) is not None
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            unsqueeze = Unsqueeze.create(graph, reshape.this, dim)
            inject_stack_trace_from(reshape, to=unsqueeze)
        return {node: ReplaceAllUses(by=unsqueeze.node)}


def find_unsqueeze_dim(
    input_shape: torch.Size,
    target_shape: torch.Size,
) -> int | None:
    ndim = len(input_shape)
    if ndim + 1 != len(target_shape):
        return None
    for i in range(ndim + 1):
        if torch.Size((*input_shape[:i], 1, *input_shape[i:])) == target_shape:
            return i
    return None
