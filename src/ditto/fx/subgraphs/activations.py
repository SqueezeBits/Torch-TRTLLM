from abc import abstractmethod

from torch.fx import Node
from typing_extensions import Self

from ..nodes import Mul, Sigmoid
from .subgraph import Subgraph


class ActivationSubgraph(Subgraph):
    @property
    @abstractmethod
    def input(self) -> Node:
        ...

    @property
    @abstractmethod
    def output(self) -> Node:
        ...


class Silu(ActivationSubgraph):
    sigmoid: Sigmoid
    mul: Mul

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (
            (mul := Mul.specialize_from(node))
            and isinstance(rhs := mul.other, Node)
            and (sigmoid := Sigmoid.specialize_from(rhs))
            and sigmoid.this == mul.this
        ):
            return cls(sigmoid=sigmoid, mul=mul)
        return None

    @property
    def input(self) -> Node:
        return self.sigmoid.this

    @property
    def output(self) -> Node:
        return self.mul.node
