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

from ...types import StrictlyTyped
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
