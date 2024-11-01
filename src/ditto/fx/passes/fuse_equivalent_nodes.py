from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from .graph_pass import GraphOptimizationPass


class FuseEquivalentNodes(GraphOptimizationPass):
    """Fuse nodes performing identical operations for the same input node."""

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        nodes = {n: i for i, n in enumerate(graph.nodes)}

        def get_closest_input_node(n: Node) -> Node:
            return max(n.all_input_nodes, key=lambda x: nodes.get(x, -1))

        modified = False
        for node in nodes:
            if not (
                groups := [
                    group for group in group_users(node) if len(group) > 1 and get_closest_input_node(group[0]) == node
                ]
            ):
                continue
            for group in groups:
                rep = group[0]
                rep.stack_trace = f"{rep.stack_trace}, pass: fused by {__name__}"
                is_replaced = [len(user.replace_all_uses_with(rep)) > 0 for user in group[1:]]
                modified = modified or any(is_replaced)
        return PassResult(graph_module, modified)


def group_users(node: Node) -> list[list[Node]]:
    users = [*node.users.keys()]
    groups: list[list[Node]] = []
    while users:
        user = users.pop(0)
        group = [user] + [other_user for other_user in users if are_equivalent(user, other_user)]
        users = [x for x in users if x not in group]
        groups.append(group)
    return groups


def are_equivalent(x: Node, y: Node) -> bool:
    return x.op == y.op and x.target == y.target and x.args == y.args and x.kwargs == y.kwargs
