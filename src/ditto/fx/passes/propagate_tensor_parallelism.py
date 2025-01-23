from enum import Enum

from torch.fx import Node

from ...configs.trtllm.pretrained import TRTLLMMapping
from ..subgraphs.linear import Linear
from ..targets import GPTAttentionPlugin
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
        prev_tp_type = get_previous_tp_type(node)

        if isinstance(node.target, GPTAttentionPlugin):
            node.target.num_heads = node.target.num_heads // self.mapping.tp_size
            node.target.num_kv_heads = (node.target.num_kv_heads + self.mapping.tp_size - 1) // self.mapping.tp_size
            node.target.tp_size = self.mapping.tp_size
            node.target.tp_rank = self.mapping.tp_rank
            node.meta["tp_type"] = prev_tp_type
            return {}
        if not Linear.configure_from(node):
            node.meta["tp_type"] = prev_tp_type
            return {}

        if prev_tp_type == TensorParallelType.NONE:
            node.meta["tp_type"] = TensorParallelType.COLUMN
        elif prev_tp_type == TensorParallelType.COLUMN:
            node.meta["tp_type"] = TensorParallelType.ROW
        else:
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
    return TensorParallelType.COLUMN if TensorParallelType.COLUMN in prev_tp_types else TensorParallelType.NONE
