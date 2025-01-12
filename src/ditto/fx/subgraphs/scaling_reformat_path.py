# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
import operator
from collections import Counter
from functools import reduce
from typing import TypeVar


import torch
from torch._ops import OpOverload
from torch.fx import Node
from typing_extensions import Self

from ...types import NodeCondition, Number
from ..utils import get_tensor_metadata
from .path import Path
from ..nodes import Mul, Div


class ScalingReformatPath(Path):    
    @classmethod
    def get_scaling_targets(cls) -> tuple[OpOverload, ...]:
        return (
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.div.Scalar,
        )
    
    @classmethod
    def get_reformat_targets(cls) -> tuple[OpOverload, ...]:
        return (
            torch.ops.aten._to_copy.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.aten.unsqueeze.default,
        )

    @classmethod
    def configure_from(cls, node: Node, *, break_if: NodeCondition = lambda _: False, max_len = 10) -> Self:
        targets = cls.get_scaling_targets() + cls.get_reformat_targets()
        return super().configure_from(
            node,
            break_if=lambda n: n.target not in targets or break_if(n),
            max_len=max_len
        )
    
    @property
    def scalings(self) -> tuple[Node]:
        return tuple(node for node in self.node_seq if node.target in self.get_scaling_targets())

    @property
    def reformats(self) -> tuple[Node, ...]:
        return tuple(node for node in self.node_seq if node.target in self.get_reformat_targets())
    
    @property
    def scale(self) -> Number:
        return reduce(
            operator.truediv, [div.other for n in self.reformats if (div := Div.specialize_from(n))],
            reduce(operator.mul, [mul.other for n in self.reformats if (mul := Mul.specialize_from(n))], 1.0)
        )

    @property
    def total_expansion(self) -> int | None:
        if not self.reformats:
            return 1
        if not ((top := get_tensor_metadata(self.reformats[-1])) and (bottom := get_tensor_metadata(self.reformats[0]))):
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
