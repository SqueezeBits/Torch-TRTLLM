# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import Literal

import torch
from loguru import logger
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
        return get_attr_reference(graph_module, name) is not None

    @property
    def parameter(self) -> torch.nn.Parameter | torch.Tensor:
        assert (graph_module := self.node.graph.owning_module) is not None
        assert (param := get_attr_reference(graph_module, self.target)) is not None
        return param

    @property
    def tensor(self) -> torch.Tensor:
        return param.data if isinstance(param := self.parameter, torch.nn.Parameter) else param


# Adapted from `get_attr_reference_exists` used in `torch.fx.Graph.get_attr`
def get_attr_reference(mod: torch.nn.Module, qualified_name: str) -> torch.Tensor | None:
    module_path, _, name = qualified_name.rpartition(".")

    try:
        submod: torch.nn.Module = mod.get_submodule(module_path)
    except AttributeError:
        logger.warning(f"Failed to fetch module {module_path}!")
        return None

    if not hasattr(submod, name):
        return None

    if isinstance(res := getattr(submod, name), torch.Tensor):
        return res

    return None
