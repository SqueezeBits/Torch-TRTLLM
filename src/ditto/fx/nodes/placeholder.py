# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import TYPE_CHECKING, Literal

import torch
from torch.fx import Graph, Node
from typing_extensions import Self

from ..utils import get_fake_mode
from .node_specialization import FinalSpecialization

if TYPE_CHECKING:
    from ...arguments import TensorTypeHint


class Placeholder(FinalSpecialization):
    @classmethod  # pylint: disable-next=arguments-differ
    def create(  # type: ignore[override]
        cls,
        graph: Graph,
        name: str,
        hint: "TensorTypeHint",
    ) -> Self:
        x = cls._specialize_from(graph.placeholder(name))
        if fake_mode := get_fake_mode(graph):
            with fake_mode:
                x.output = torch.empty(hint.symbolic_shape, dtype=hint.dtype)  # type: ignore[assignment]
        return x

    @property
    def target(self) -> str:
        assert isinstance(name := super().target, str)
        return name

    @classmethod
    def designated_op(cls) -> Literal["placeholder"]:
        return "placeholder"

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return super().validate_node(node) and isinstance(node.target, str)
