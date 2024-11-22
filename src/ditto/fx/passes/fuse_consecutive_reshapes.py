from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult

from ..nodes import Reshape, Unsqueeze
from .graph_pass import GraphOptimizationPass


class FuseConsecutiveReshapes(GraphOptimizationPass):
    """Fuse two consecutive reshapes."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not (
                (_ := Reshape.specialize_from(node) or Unsqueeze.specialize_from(node))
                and (
                    children := [
                        child_reshape
                        for child_node in node.users
                        if (child_reshape := Reshape.specialize_from(child_node))
                    ]
                )
            ):
                continue
            for child_reshape in children:
                child_node = child_reshape.node
                child_node.replace_input_with(node, node.all_input_nodes[0])
                child_node.stack_trace = f"{child_node.stack_trace}, pass: fused with {node} by {__name__}"
                modified = True
        return PassResult(graph_module, modified)
