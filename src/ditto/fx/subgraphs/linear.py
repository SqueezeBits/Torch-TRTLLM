from torch.fx import Node
from typing_extensions import Self

from ..nodes import MM, Add, GetAttr, Reshape
from .subgraph import Subgraph


class WeightMatmul(Subgraph):
    mm: MM
    weight: GetAttr

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (mm := MM.specialize_from(node)) and (weight := GetAttr.specialize_from(mm.other)):
            return cls(mm=mm, weight=weight)
        return None


class BiasAdd(Subgraph):
    add: Add
    bias: GetAttr

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (
            (add := Add.specialize_from(node))
            and isinstance(add.other, Node)
            and (bias := GetAttr.specialize_from(add.other))
            and bias.parameter.ndim == 1
        ):
            return cls(add=add, bias=bias)
        return None


class Linear(Subgraph):
    weight_matmul: WeightMatmul
    bias_add: BiasAdd | None

    @property
    def mm(self) -> MM:
        return self.weight_matmul.mm

    @property
    def weight(self) -> GetAttr:
        return self.weight_matmul.weight

    @property
    def has_bias(self) -> bool:
        return self.bias_add is not None

    @property
    def add(self) -> Add:
        assert self.bias_add is not None
        return self.bias_add.add

    @property
    def bias(self) -> GetAttr:
        assert self.bias_add is not None
        return self.bias_add.bias

    @property
    def input(self) -> Node:
        return self.mm.this

    @property
    def output(self) -> Node:
        return self.add.node if self.has_bias else self.mm.node

    @property
    def reshape_in(self) -> Reshape | None:
        return Reshape.specialize_from(self.mm.this)

    @property
    def reshape_out(self) -> Reshape | None:
        return Reshape.specialize_from(self.output)

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (weight_matmul := WeightMatmul.configure_from(node)):
            return None
        bias_add = None if len(users := list(weight_matmul.mm.users)) != 1 else BiasAdd.configure_from(users[0])
        if bias_add is not None and bias_add.bias.parameter.shape[-1] != weight_matmul.weight.parameter.shape[-1]:
            bias_add = None
        return cls(weight_matmul=weight_matmul, bias_add=bias_add)

    # @classmethod
    # def configure_from(cls, node: Node) -> Self | None:
    #     if (
    #         (mm := MM.specialize_from(node))
    #         and len(mm.users) == 1
    #     ):
    #         Add.specialize_from([*node.users][0])
    #         return cls(
    #             mm_const=mm_const,
    #             input_reshape=input_reshape,
    #             output_add_or_reshape=output_add_or_reshape,
    #         )
    #     return None

    # @property
    # def input_node(self) -> Node:
    #     return self.input_reshape.this
