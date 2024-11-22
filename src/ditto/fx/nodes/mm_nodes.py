# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.node import Node

from .call_function_node import CallFunctionNode


class MMNode(CallFunctionNode):
    lhs: Node
    rhs: Node

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.mm.default,)


class MMConstNode(MMNode):
    @classmethod
    def validate_node(cls, node: Node) -> bool:
        if (
            super().validate_node(node)
            and (rhs := node.all_input_nodes[1]).op == "get_attr"
            and isinstance(target := rhs.target, str)
            and (graph_module := rhs.graph.owning_module)
        ):
            try:
                _ = graph_module.get_parameter(target)
                return True
            except AttributeError:
                pass
        return False

    @property
    def weight_name(self) -> str:
        assert isinstance(target := self.rhs.target, str)
        return target

    @property
    def weight(self) -> torch.nn.Parameter:
        assert (graph_module := self.rhs.graph.owning_module)
        return graph_module.get_parameter(self.weight_name)
