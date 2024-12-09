# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import Literal

from torch.fx.node import Node

from .node_specialization import NodeSpecialization


class Placeholder(NodeSpecialization):
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
