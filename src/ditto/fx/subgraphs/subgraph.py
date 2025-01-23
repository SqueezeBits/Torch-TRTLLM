from abc import ABC, abstractmethod

from torch.fx import Node
from typing_extensions import Self

from ...types import StrictlyTyped
from ..nodes.node_specialization import NodeSpecialization


class Subgraph(StrictlyTyped, ABC):
    @classmethod
    @abstractmethod
    def configure_from(cls, node: Node) -> Self | None:
        ...

    @property
    def nodes(self) -> list[Node]:
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
            for self_node, other_node in zip(self.nodes, other.nodes):
                if self_node is not other_node:
                    return False
            return True
        return False
