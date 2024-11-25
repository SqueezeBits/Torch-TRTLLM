from torch.fx import Node
from typing_extensions import Self

from ..nodes import MM, GetAttr
from .subgraph import Subgraph


class MMConst(Subgraph):
    mm: MM
    weight: GetAttr

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (mm := MM.specialize_from(node)) and (weight := GetAttr.specialize_from(mm.other)):
            return cls(mm=mm, weight=weight)
        return None
