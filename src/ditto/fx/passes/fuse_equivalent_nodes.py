# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.fx import Node

from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class FuseEquivalentNodes(NodewiseOptimizationPass):
    """Fuse nodes performing identical operations for the same input node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (groups := [group for group in group_users(node) if len(group) > 1]):
            return {}

        results: dict[Node, NodewisePassResult] = {}
        for group in groups:
            rep = group[0]
            results.update({user: ReplaceAllUses(by=rep) for user in group[1:]})
        return results


def group_users(node: Node) -> list[list[Node]]:
    """Group users of a node that perform equivalent operations.

    Args:
        node (Node): The node whose users should be grouped

    Returns:
        list[list[Node]]: A list of groups, where each group is a list of equivalent nodes.
        The first node in each group is considered the representative node.
    """
    users = [*node.users.keys()]
    groups: list[list[Node]] = []
    while users:
        user = users.pop(0)
        group = [user] + [other_user for other_user in users if are_equivalent(user, other_user)]
        users = [x for x in users if x not in group]
        groups.append(group)
    return groups


def are_equivalent(x: Node, y: Node) -> bool:
    """Check if two nodes perform equivalent operations.

    Args:
        x (Node): First node to compare
        y (Node): Second node to compare

    Returns:
        bool: True if the nodes have identical operation, target, arguments and keyword arguments
    """
    return x.op == y.op and x.target == y.target and x.args == y.args and x.kwargs == y.kwargs
