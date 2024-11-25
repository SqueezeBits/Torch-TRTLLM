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
        return [
            attr.node
            for name in self.model_dump().keys()
            if isinstance((attr := getattr(self, name, None)), NodeSpecialization)
        ]

    def __hash__(self) -> int:
        assert (
            node_hashes := [hash(node) for node in self.nodes]
        ), f"{type(self).__name__} does not have specialized node attributes."
        return sum(node_hashes)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Subgraph):
            return self is other
        return False
