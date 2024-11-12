import torch
from loguru import logger
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult

from .graph_pass import GraphOptimizationPass


class MakeWeightsContiguous(GraphOptimizationPass):
    """Make non-contiguous weights contiguous."""

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for name, param in graph_module.named_parameters():
            if param.is_contiguous():
                continue
            setattr(graph_module, name, torch.nn.Parameter(param.contiguous()))
            logger.debug(f"{self.indent}  Made the parameter {name} contiguous")
            modified = True
        return PassResult(graph_module, modified)
