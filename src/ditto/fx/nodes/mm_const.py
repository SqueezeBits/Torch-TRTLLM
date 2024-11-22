# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch._ops import OpOverload
from torch.fx.node import Node

from .aten import MM


class MMConst(MM):
    @classmethod
    def possible_targets(cls) -> tuple[OpOverload, ...]:
        return (torch.ops.aten.mm.default,)

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
        assert isinstance(target := self.other.target, str)
        return target

    @property
    def weight(self) -> torch.nn.Parameter:
        assert (graph_module := self.other.graph.owning_module)
        return graph_module.get_parameter(self.weight_name)
