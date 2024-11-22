from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult

from ..nodes import ToCopyNode
from .graph_pass import GraphOptimizationPass


class FuseConsecutiveToCopys(GraphOptimizationPass):
    """Fuse two consecutive _to_copy nodes."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not (
                (parent := ToCopyNode.specialize_from(node))
                and (
                    children := [
                        child for child_node in node.users if (child := ToCopyNode.specialize_from(child_node))
                    ]
                )
            ):
                continue
            for child in children:
                child_node = child.node
                child_node.replace_input_with(node, parent.x)
                if stack_trace := child_node.stack_trace:
                    child_node.stack_trace = f"{stack_trace}, pass: fused with {node} by {__name__}"
        return PassResult(graph_module, modified)
