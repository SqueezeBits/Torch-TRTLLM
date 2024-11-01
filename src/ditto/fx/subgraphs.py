from abc import ABC, abstractmethod

import torch
from torch.fx import Node
from typing_extensions import Self

from ..types import StrictlyTyped


class Subgraph(StrictlyTyped, ABC):
    @classmethod
    @abstractmethod
    def configure_from(cls, node: Node) -> Self | None:
        ...

    def __hash__(self) -> int:
        assert (node_hashes := [hash(attr) for attr in self.model_dump().values() if isinstance(attr, Node)])
        return sum(node_hashes)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Subgraph):
            return self is other
        return False


class LinearSubgraph(Subgraph):
    mm: Node
    weight: Node
    input_reshape: Node
    output_reshape: Node

    @classmethod
    def configure_from(cls, mm: Node) -> Self | None:
        if (
            mm.op == "call_function"
            and mm.target is torch.ops.aten.mm.default
            and len(mm.all_input_nodes) == 2
            and (input_reshape := mm.all_input_nodes[0]).op == "call_function"
            and input_reshape.target is torch.ops.aten.reshape.default
            and (weight := mm.all_input_nodes[1]).op == "get_attr"
            and len(mm.users) == 1
            and (output_reshape := [*mm.users][0]).op == "call_function"
            and output_reshape.target is torch.ops.aten.reshape.default
        ):
            return cls(
                mm=mm,
                weight=weight,
                input_reshape=input_reshape,
                output_reshape=output_reshape,
            )
        return None

    @property
    def input_tensor(self) -> Node:
        return self.input_reshape.all_input_nodes[0]

    @property
    def users(self) -> dict[Node, None]:
        return self.output_reshape.users
