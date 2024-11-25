# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import Literal

import torch
from torch.fx.node import Node

from .node_specialization import NodeSpecialization


class GetAttr(NodeSpecialization):
    @property
    def target(self) -> str:
        assert isinstance(name := super().target, str)
        return name

    @classmethod
    def designated_op(cls) -> Literal["get_attr"]:
        return "get_attr"

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        if not (
            super().validate_node(node)
            and (graph_module := node.graph.owning_module)
            and isinstance(name := node.target, str)
        ):
            return False
        try:
            _ = graph_module.get_parameter(name)
            return True
        except AttributeError:
            return False

    @property
    def parameter(self) -> torch.nn.Parameter:
        assert (graph_module := self.node.graph.owning_module) is not None
        return graph_module.get_parameter(self.target)
