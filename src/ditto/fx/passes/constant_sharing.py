import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from .graph_pass import GraphOptimizationPass


class ConstantSharing(GraphOptimizationPass):
    """Reuse constant attributes sharing the same values."""

    def call(self, graph_module: GraphModule) -> PassResult:
        get_attrs = [n for n in graph_module.graph.nodes if n.op == "get_attr"]
        value_cache: dict[torch.Size, dict[Node, torch.nn.Parameter]] = {}
        modified = False
        for node in get_attrs:
            if not isinstance(node.target, str):
                continue
            try:
                value = graph_module.get_parameter(node.target)
            except AttributeError:
                continue
            if value.shape not in value_cache:
                value_cache[value.shape] = {node: value}
                continue
            existing_values = value_cache[value.shape]
            for existing_node, existing_value in existing_values.items():
                if torch.all(existing_value == value):
                    node.replace_all_uses_with(existing_node)
                    modified = True
                    break
            else:
                existing_values[node] = value
        return PassResult(graph_module, modified)
