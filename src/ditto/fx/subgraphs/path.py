# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import TypeVar

import torch
from pydantic import model_validator
from torch._ops import OpOverload
from torch.fx import Node
from typing_extensions import Self

from ...types import NodeCondition
from ..nodes import NodeSpecialization
from ..utils import get_tensor_metadata
from .subgraph import Subgraph


class Path(Subgraph):
    node_seq: tuple[Node, ...]  # sorted from bottom to top

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
            if top in nodes:
                break

            nodes.append(top)
            if not all(cond(top) for cond in [has_single_parent, lambda n: not break_if(n)]) or len(nodes) == max_len:
                break

            top = top.all_input_nodes[0]

        return cls(node_seq=tuple(nodes))

    @model_validator(mode="after")  # type: ignore[misc]
    def validate_adjacency(self) -> Self:
        assert self.nodes
        assert len(set(self.nodes)) == len(self.nodes)

        for i, node in enumerate(self.nodes[:-1]):
            assert has_single_parent(node)
            assert self.nodes[i + 1] == node.all_input_nodes[0]

        return self


def has_single_parent(node: Node) -> bool:
    return len([n for n in node.all_input_nodes if n.target is not torch.ops.aten.sym_size.int]) == 1


# pylint: disable-next=invalid-name
NodeType = TypeVar("NodeType", bound=NodeSpecialization)


class TrailingReformatPath(Path):
    @property
    def reformats(self) -> tuple[Node, ...]:
        return self.node_seq[:-1]

    @classmethod
    def get_reformat_targets(cls) -> tuple[OpOverload, ...]:
        return (
            torch.ops.aten._to_copy.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.permute.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.aten.unsqueeze.default,
        )

    @classmethod
    def configure_from(cls, node: Node, *, break_if: NodeCondition = lambda _: False, max_len: int = -1) -> Self:
        return super().configure_from(
            node, break_if=lambda n: n.target not in cls.get_reformat_targets() or break_if(n), max_len=max_len
        )

    @property
    def total_expansion(self) -> int | None:
        if not self.reformats:
            return 1
        if not ((top := get_tensor_metadata(self.top)) and (bottom := get_tensor_metadata(self.bottom))):
            return None
        # Note that `torch.ops.aten.expand.default` is the only target that can increase the number of elements.
        # A naive implementation would be simple:
        # `return bottom.shape.numel() // top.shape.numel()`
        # However, this naive implementation involves direct division on `torch.SymInt` objects,
        # adding unwanted shape constraints to the ambient shape environment.
        # As a result, it can affect the `FakeTensorProp` to believe some of the `torch.SymInt` objects are constant.
        # Therefore, we must handle `int` and `torch.SymInt` objects separately here.
        bottom_sym_shape = [s for s in bottom.shape if isinstance(s, torch.SymInt)]
        top_sym_shape = [s for s in top.shape if isinstance(s, torch.SymInt)]
        if are_same_as_sets(bottom_sym_shape, top_sym_shape):
            return None

        bottom_ishape = torch.Size(s for s in bottom.shape if isinstance(s, int))
        top_ishape = torch.Size(s for s in top.shape if isinstance(s, int))
        return bottom_ishape.numel() // top_ishape.numel()

    @classmethod
    def traceback(
        cls,
        node_type: type[NodeType],
        node: Node,
        *,
        break_if: NodeCondition = lambda _: False,
    ) -> NodeType | None:
        return node_type.specialize_from(TrailingReformatPath.configure_from(node, break_if=break_if).top)


T = TypeVar("T")


def are_same_as_sets(one: list[T], another: list[T]) -> bool:
    """Check if two lists of objects consists of the same set of elements.

    Note: this is a workaround for `set(one) == set(another)`, which does not work
        if `T` is not a hashable type, for example, `torch.SymInt`.

    Args:
        one (list[T]): a list of objects
        another (list[T]): another list containing the objects of the same type.

    Returns:
        bool: True if two lists consists of the same set of elements, False otherwise.
    """
    if len(one) != len(another):
        return False
    one = one[:]
    another = another[:]
    while one:
        s = one.pop()
        matched_idx: int
        for i, t in enumerate(another):
            if s is t:
                matched_idx = i
                break
        else:
            return False
        another.pop(matched_idx)
    return True
