from torch.fx import Node

from ..nodes import Reshape
from ..utils import get_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass


class EliminateNopReshape(NodeWiseOptimizationPass):
    """Eliminate reshape whose target shape is identical to the input shape."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (reshape := Reshape.specialize_from(node))
            and (input_tensor := get_tensor_metadata(reshape.this))
            and (output_tensor := get_tensor_metadata(reshape.node))
            and input_tensor.shape == output_tensor.shape
        ):
            return {}
        return {node: reshape.this}
