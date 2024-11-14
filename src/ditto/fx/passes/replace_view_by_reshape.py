import torch
from torch.fx import Node

from .node_wise_pass import NodeWiseOptimizationPass


class ReplaceViewByReshape(NodeWiseOptimizationPass):
    """A replacement for the `view_to_reshape` pass in TorchTRT for its shape inference error."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (node.op == "call_function" and node.target is torch.ops.aten.view.default):
            return {}
        node.target = torch.ops.aten.reshape.default
        return {node: node}
