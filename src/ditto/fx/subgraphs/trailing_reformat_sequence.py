# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from collections.abc import Callable
from typing import TypeVar

import torch
from pydantic import model_validator
from torch._ops import OpOverload
from torch.fx import Node
from typing_extensions import Self

from ..nodes import NodeSpecialization
from ..utils import get_tensor_metadata
from .subgraph import Subgraph

# pylint: disable-next=invalid-name
NodeType = TypeVar("NodeType", bound=NodeSpecialization)


class TrailingReformatSequence(Subgraph):
    node_seq: tuple[Node, ...]  # sorted from bottom to top

    @property
    def nodes(self) -> list[Node]:
        return list(self.node_seq)

    @property
    def top(self) -> Node:
        return self.node_seq[-1]

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
    def configure_from(cls, node: Node, *, break_if: Callable[[Node], bool] | None = None) -> Self:
        nodes: list[Node] = []
        top = node
        targets = cls.get_reformat_targets()
        while top.target in targets:
            nodes.append(top)
            top = top.all_input_nodes[0]
            if break_if is not None and break_if(top):
                break
        nodes.append(top)
        return cls(node_seq=tuple(nodes))

    @model_validator(mode="after")
    def validate_adjacency(self) -> Self:
        assert len(self.node_seq) > 0
        targets = self.get_reformat_targets()
        for i in range(len(self.node_seq) - 1):
            reformat_node = self.node_seq[i]
            assert reformat_node.target in targets
            next_node = self.node_seq[i + 1] if i + 1 < len(self.node_seq) else self.top
            assert reformat_node.all_input_nodes[0] == next_node
        return self

    @property
    def total_expansion(self) -> int | None:
        if not self.reformats:
            return 1
        if not ((top := get_tensor_metadata(self.top)) and (bottom := get_tensor_metadata(self.reformats[0]))):
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
        break_if: Callable[[Node], bool] | None = None,
    ) -> NodeType | None:
        return node_type.specialize_from(TrailingReformatSequence.configure_from(node, break_if=break_if).top)


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
