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

from ..nodes import (
    SafeSoftmax,
    SoftmaxDefault,
)
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class ReplaceSafeSoftmaxBySoftmax(NodewiseOptimizationPass):
    """Replace safe softmax node by softmax node.

    This is a workaround for that aten._safe_softmax is not supported by torch-tensorrt.
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (safe_softmax := SafeSoftmax.specialize_from(node)):
            return {}

        with node.graph.inserting_after(node):
            softmax = SoftmaxDefault.create(
                node.graph,
                safe_softmax.this,
                safe_softmax.dim,
                safe_softmax.dtype == torch.float32 if safe_softmax.dtype is not None else False,
            )

        return {node: ReplaceAllUses(by=softmax.node)}
