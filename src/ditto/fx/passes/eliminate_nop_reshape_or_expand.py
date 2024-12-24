from torch.fx import Node

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
            and input_tensor.shape == output_tensor.shape
        ):
            return {}
        return {node: ReplaceAllUses(by=reshape.this)}
