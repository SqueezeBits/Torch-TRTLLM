from torch.fx import GraphModule, Node

from ditto.fx.passes.node_wise_pass import NodewisePassResult

from .node_wise_pass import NodewiseOptimizationPass, ReplaceAllUses


class FuseEquivalentNodes(NodewiseOptimizationPass):
    """Fuse nodes performing identical operations for the same input node."""

    def __init__(self, *, depth: int = 0) -> None:
        super().__init__(depth=depth)
        self.nodes: dict[Node, int] | None = None

    def requires(self, graph_module: GraphModule) -> None:
        self.nodes = {n: i for i, n in enumerate(graph_module.graph.nodes)}

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            groups := [
                group for group in group_users(node) if len(group) > 1 and self.get_closest_input_node(group[0]) == node
            ]
        ):
            return {}

        results: dict[Node, NodewisePassResult] = {}
        for group in groups:
            rep = group[0]
            rep.stack_trace = f"{rep.stack_trace}, pass: fused by {__name__}"
            results.update({user: ReplaceAllUses(by=rep) for user in group[1:]})
        return results

    def ensures(self, graph_module: GraphModule) -> None:
        self.nodes = None

    def get_closest_input_node(self, n: Node) -> Node:
        assert (nodes := self.nodes) is not None
        return max(n.all_input_nodes, key=lambda x: nodes.get(x, -1))


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
