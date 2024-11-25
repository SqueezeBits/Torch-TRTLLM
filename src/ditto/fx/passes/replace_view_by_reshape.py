import torch
from torch.fx import Node

from .node_wise_pass import ModifiedInsideThePass, NodewiseOptimizationPass, NodewisePassResult


class ReplaceViewByReshape(NodewiseOptimizationPass):
    """A replacement for the `view_to_reshape` pass in TorchTRT for its shape inference error."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (node.op == "call_function" and node.target is torch.ops.aten.view.default):
            return {}
        node.target = torch.ops.aten.reshape.default
        return {node: ModifiedInsideThePass()}
