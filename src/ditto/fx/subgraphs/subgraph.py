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

from abc import ABC, abstractmethod

from torch.fx import Graph, Node
from typing_extensions import Self

from ...types import NodeCriterion, StrictlyTyped
from ..nodes.node_specialization import NodeSpecialization


class Subgraph(StrictlyTyped, ABC):
    """Base class for identifying and extracting subgraph patterns in a computation graph.

    A subgraph represents a pattern of connected nodes in a computation graph that form a
    logical unit, such as a linear layer or an attention block. This class provides methods
    to find and extract these patterns from a PyTorch FX graph.

    The class is abstract and requires subclasses to implement the configure_from method
    to define their specific pattern matching logic.

    Methods:
        configure_from: Abstract method to extract configuration from a node
        find_nearest: Find nearest matching subgraph by traversing ancestors or descendants
        find_last: Find last occurrence of subgraph in computation graph
        nodes: Property that returns all nodes in the subgraph
    """

    @classmethod
    @abstractmethod
    def configure_from(cls, node: Node) -> Self | None:
        """Extract configuration from a node.

        Args:
            node (Node): The node to extract configuration from

        Returns:
            Self | None: The subgraph configuration if found, None otherwise
        """

    @classmethod
    def find_nearest(
        cls,
        from_node: Node,
        follow_parent: bool = True,
        follow_first_only: bool = True,
        break_if: NodeCriterion | None = None,
    ) -> Self | None:
        """Find the nearest occurrence of the subgraph by traversing the node's ancestors or descendants.

        Performs a breadth-first search through either ancestor nodes or descendant nodes and finds
        the first subgraph that matches the pattern, which will be the closest to the given node
        in terms of graph traversal depth.

        Args:
            from_node (Node): Starting node to search from
            follow_parent (bool): If True, search through ancestor nodes. If False, search through
                descendant nodes. Defaults to True.
            follow_first_only (bool): If True, only follow the first node in the parent or child
                list. If False, follow all nodes in the list. Defaults to True.
            break_if (NodeCriterion | None): If provided, stop searching if the node matches the criterion

        Returns:
            Self | None: The nearest matching subgraph if one exists, None otherwise
        """
        queue = [from_node]
        while queue:
            node = queue.pop(0)
            if subgraph := cls.configure_from(node):
                return subgraph
            if break_if is not None and break_if(node):
                break
            if not (next_nodes := list(node.all_input_nodes if follow_parent else node.users)):
                continue
            if follow_first_only:
                queue.append(next_nodes[0])
            else:
                queue.extend(next_nodes)
        return None

    @classmethod
    def find_last(cls, graph: Graph) -> Self | None:
        """Find the last occurrence of the subgraph in the computation graph.

        Searches through all nodes in the graph and finds the last subgraph that is closest
        to the end of the graph in terms of graph traversal depth.

        Args:
            graph (Graph): The computation graph to search in

        Returns:
            Self | None: The last subgraph if found, None otherwise
        """
        for node in reversed(graph.nodes):
            if subgraph := cls.configure_from(node):
                return subgraph
        return None

    @property
    def nodes(self) -> list[Node]:
        """All nodes in the subgraph."""
        all_nodes: list[Node] = []
        for name in self.model_dump():
            if isinstance((attr := getattr(self, name, None)), NodeSpecialization):
                all_nodes.append(attr.node)
            elif isinstance(attr, Node):
                all_nodes.append(attr)
            elif isinstance(attr, Subgraph):
                all_nodes.extend(attr.nodes)
        return all_nodes

    def __hash__(self) -> int:
        assert (
            node_hashes := [hash(node) for node in self.nodes]
        ), f"{type(self).__name__} does not have specialized node attributes."
        return sum(node_hashes)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Subgraph) and len(self.nodes) == len(other.nodes):
            return all(x is y for x, y in zip(self.nodes, other.nodes))
        return False
