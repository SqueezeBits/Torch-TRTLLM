from loguru import logger
from torch.fx import GraphModule
from torch.fx.graph import dtype_abbrs
from torch.fx.passes.infra.pass_base import PassResult

from .graph_pass import GraphOptimizationPass


class EliminateUnusedWeights(GraphOptimizationPass):
    """Eliminate unused weights from the graph module."""

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        used_weight_names = [
            weight_name
            for node in graph_module.graph.nodes
            if node.op == "get_attr" and isinstance(weight_name := node.target, str)
        ]
        unused_weights = {
            name: param for name, param in graph_module.named_parameters() if name not in used_weight_names
        }
        for name, param in unused_weights.items():
            shape = ", ".join(str(s) for s in param.shape)
            dtype = dtype_abbrs[param.dtype]
            if hasattr(graph_module, name):
                delattr(graph_module, name)
                logger.debug(f"{self.indent}  Removed the unused parameter {name} {dtype}[{shape}]")
        return PassResult(graph_module, modified)
