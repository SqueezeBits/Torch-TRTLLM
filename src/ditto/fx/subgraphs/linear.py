from torch.fx import Node
from typing_extensions import Self

from ..nodes import Add, Reshape
from .mm_const import MMConst
from .subgraph import Subgraph


class Linear(Subgraph):
    mm_const: MMConst
    input_reshape: Reshape
    output_add_or_reshape: Add | Reshape

    @property
    def mm(self) -> Node:
        return self.mm_const.mm.node

    @property
    def output(self) -> Node:
        if isinstance(self.output_add_or_reshape, Reshape):
            return self.mm_const.mm.node
        return self.output_add_or_reshape.node

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (
            (mm_const := MMConst.configure_from(node))
            and len(mm_const.mm.users) == 1
            and (input_reshape := Reshape.specialize_from(mm_const.mm.this))
            and (
                output_add_or_reshape := (
                    Reshape.specialize_from([*node.users][0]) or Add.specialize_from([*node.users][0])
                )
            )
        ):
            return cls(
                mm_const=mm_const,
                input_reshape=input_reshape,
                output_add_or_reshape=output_add_or_reshape,
            )
        return None

    @property
    def input_node(self) -> Node:
        return self.input_reshape.this
