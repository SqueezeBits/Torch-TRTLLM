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

from enum import Enum

from torch.fx import Node

from ...configs.trtllm.pretrained import TRTLLMMapping
from ..metadata_keys import EXPERT_TYPE
from ..nodes import MM
from .infra import NodewiseOptimizationPass, NodewisePassResult


class TensorParallelType(Enum):
    """Tensor parallel type.

    Attributes:
        NONE : The tensor parallel type is none
        COLUMN : The tensor parallel type is column
        ROW : The tensor parallel type is row
    """

    NONE = "none"
    COLUMN = "column"
    ROW = "row"

    def __str__(self) -> str:
        return self.name


class PropagateTensorParallelism(NodewiseOptimizationPass):
    """Propagate tensor parallelism in the graph.

    It propagates the properties of TP for some plugin nodes
    and the tensor parallel type in the path of the current node according to the following rules:
    - If the current node is a linear node and the previous tensor parallel type is TensorParallelType.NONE,
      it will set the tensor parallel type of this node to TensorParallelType.COLUMN.
    - If the current node is a linear node and the previous tensor parallel type is TensorParallelType.COLUMN,
      it will set the tensor parallel type of this node to TensorParallelType.ROW.
    - The previous tensor parallel type is TensorParallelType.ROW is not valid, so the previous tensor parallel type
      is always TensorParallelType.NONE or TensorParallelType.COLUMN.
    - If the current node is not a linear node, it just propagates the tensor parallel type of the previous node.

    Attributes:
        mapping (TRTLLMMapping): The mapping of the model
    """

    mapping: TRTLLMMapping

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if self.mapping.tp_size == 1:
            return {}
        prev_tp_type = get_previous_tp_type(node)

        if not MM.specialize_from(node):
            node.meta["tp_type"] = prev_tp_type
            return {}

        if prev_tp_type == TensorParallelType.COLUMN:
            node.meta["tp_type"] = TensorParallelType.ROW
        else:
            node.meta["tp_type"] = TensorParallelType.COLUMN

        if should_exclude_from_tp(node):
            node.meta["tp_type"] = TensorParallelType.NONE

        return {}


def get_previous_tp_type(node: Node) -> TensorParallelType:
    """Get the previous parallel linear type of the node.

    Args:
        node (Node): The node to get the previous parallel linear type from

    Returns:
        TensorParallelType: The previous parallel linear type
    """
    if len(node.all_input_nodes) == 0:
        return TensorParallelType.NONE
    prev_tp_types: list[TensorParallelType] = []
    for prev_node in node.all_input_nodes:
        prev_tp_types.append(prev_node.meta.get("tp_type", TensorParallelType.NONE))
    # TODO: Check if it's safe to return COLUMN when there are both COLUMN and ROW in prev_tp_types.
    return TensorParallelType.COLUMN if TensorParallelType.COLUMN in prev_tp_types else TensorParallelType.NONE


def should_exclude_from_tp(node: Node) -> bool:
    """Check if a node should be excluded from tensor parallelism.

    It excludes router nodes from tensor parallelism.

    Args:
        node (Node): The node to check

    Returns:
        bool: True if the node should be excluded from tensor parallelism, False otherwise
    """
    assert MM.specialize_from(node)
    if node.meta.get(EXPERT_TYPE) in ("router", "shared_expert_gate"):
        return True
    return False
