import operator

import torch
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult

from .graph_pass import GraphOptimizationPass


class ReplaceOperatorSubByATenSub(GraphOptimizationPass):
    """Replace operator.sub by torch.ops.aten.sub op (required for torch-trt)."""

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if not (node.target is operator.sub and len(node.args) == 2):
                continue
            lhs, rhs = node.args
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                node.target = torch.ops.aten.sub.Tensor
            elif isinstance(lhs, torch.Tensor) and isinstance(rhs, bool | complex | float | int):
                node.target = torch.ops.aten.sub.Scalar
            elif isinstance(lhs, bool | complex | float | int) and isinstance(rhs, torch.Tensor):
                node.target = torch.ops.aten.sub.Scalar
                node.args = node.args[::-1]
            elif isinstance(lhs, int) and isinstance(rhs, int):
                node.target = torch.ops.aten.sub.int
            elif isinstance(lhs, float) and isinstance(rhs, float):
                node.target = torch.ops.aten.sub.float
            elif isinstance(lhs, float) and isinstance(rhs, complex):
                node.target = torch.ops.aten.sub.float_complex
            elif isinstance(lhs, complex) and isinstance(rhs, float):
                node.target = torch.ops.aten.sub.complex_float
            elif isinstance(lhs, int) and isinstance(rhs, float):
                node.target = torch.ops.aten.sub.int_float
            elif isinstance(lhs, float) and isinstance(rhs, int):
                node.target = torch.ops.aten.sub.float_int
            else:
                node.target = torch.ops.aten.sub.default
            modified = True
        return PassResult(graph_module, modified)
