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
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

from ...contexts import set_logger_level
from ..nodes import Expand, Reshape
from ..utils import get_tensor_metadata
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class EliminateNopReshapeOrExpand(NodewiseOptimizationPass):
    """Eliminate reshape whose target shape is identical to the input shape."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (reshape := Reshape.specialize_from(node) or Expand.specialize_from(node))
            and (input_tensor := get_tensor_metadata(reshape.this))
            and (output_tensor := get_tensor_metadata(reshape.node))
        ):
            return {}

        with set_logger_level("torch.fx.experimental.recording", "CRITICAL"):
            try:
                if not input_tensor.shape == output_tensor.shape:
                    return {}
            except GuardOnDataDependentSymNode:
                # NOTE: Comparing shapes between an unhinted symbolic shape and a concrete shape
                #       raises GuardOnDataDependentSymNode. If this exception occurs, it means
                #       that the shapes of input and output are different, so we don't eliminate
                #       the node.
                #
                # Examples:
                #   torch.Size([u120]) == torch.Size([1])     -> Exception
                #   torch.Size([u120]) == torch.Size([u120])  -> True
                return {}
        return {node: ReplaceAllUses(by=reshape.this)}
