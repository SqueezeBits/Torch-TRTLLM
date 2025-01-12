# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
from collections.abc import Callable

import torch
from pydantic import model_validator
from torch.fx import Node
from typing_extensions import Self

from ...types import NodeCondition
from .subgraph import Subgraph


class Path(Subgraph):
    node_seq: tuple[Node, ...] # sorted from bottom to top

    @property
    def nodes(self) -> list[Node]:
        return list(self.node_seq)

    @property
    def top(self) -> Node:
        return self.node_seq[-1]

    @property
    def bottom(self) -> Node:
        return self.node_seq[0]

    @classmethod
    def configure_from(cls, node: Node, *, break_if: NodeCondition = lambda _: False, max_len: int = 10) -> Self:
        nodes: list[Node] = []
        top = node

        while True:
            if top in nodes: # a path should be acyclic
                break

            nodes.append(top)
            if not all(cond(top) for cond in [
                has_single_parent,
                lambda n: not break_if(n)
            ]) or len(nodes) == max_len:
                break
            
            top = top.all_input_nodes[0]

        return cls(node_seq=tuple(nodes))

    @model_validator(mode="after")
    def validate_adjacency(self) -> Self:
        # TODO
        # assert len(self.node_seq) > 0
        # for i in range(len(self.node_seq)):
        #     node = self.node_seq[i]
        #     assert has_single_parent(node)

        #     if i + 1 < len(self.node_seq):
        #         next_node = self.node_seq[i + 1]
        #         assert node.all_input_nodes[0] == next_node

        return self


def has_single_parent(node: Node) -> bool:
    return len(list(n for n in node.all_input_nodes if n.target is not torch.ops.aten.sym_size.int))