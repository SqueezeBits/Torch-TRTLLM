from torch.fx import Node

from ..utils import get_tensor_metadata
from .node_wise_pass import NodeWiseOptimizationPass
from .specialized_node import ReshapeNode


class EliminateNopReshape(NodeWiseOptimizationPass):
    """Eliminate reshape whose target shape is identical to the input shape."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (reshape := ReshapeNode.specialize_from(node))
            and (input_tensor := get_tensor_metadata(reshape.x))
            and (target_shape := reshape.target_shape)
            and target_shape == input_tensor.shape
        ):
            return {}
        return {node: reshape.x}
