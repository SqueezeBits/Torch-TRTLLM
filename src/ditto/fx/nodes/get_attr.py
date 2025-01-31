# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from typing import Literal

import torch
from loguru import logger
from torch.fx import Graph, Node
from typing_extensions import Self

from ..metadata_keys import ORIGINAL_TARGET
from .node_specialization import FinalSpecialization


class GetAttr(FinalSpecialization):
    """Specialization for get_attr nodes that access tensor attributes.

    This class handles nodes that retrieve tensor attributes (parameters or buffers) from modules.
    """

    @classmethod  # pylint: disable-next=arguments-differ
    def create(cls, graph: Graph, target: str, value: torch.Tensor) -> Self:
        if graph_module := graph.owning_module:
            set_attr_reference(graph_module, target, value)
        x = cls._specialize_from(graph.get_attr(target))
        return x

    @property
    def target(self) -> str:
        assert isinstance(name := super().target, str)
        return name

    @property
    def original_target(self) -> str:
        """The original target of the get_attr node."""
        assert isinstance(target := self.meta.get(ORIGINAL_TARGET, self.target), str)
        return target

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
        """The parameter or tensor attribute accessed by this node."""
        assert (graph_module := self.node.graph.owning_module) is not None
        assert (param := get_attr_reference(graph_module, self.target)) is not None
        return param

    @property
    def tensor(self) -> torch.Tensor:
        """The attribute accessed by this node converted as a tensor."""
        return param.data if isinstance(param := self.parameter, torch.nn.Parameter) else param


def get_attr_reference(mod: torch.nn.Module, qualified_name: str) -> torch.Tensor | None:
    """Get a tensor attribute from a module by its qualified name.

    Adapted from `get_attr_reference_exists` used in `torch.fx.Graph.get_attr`

    Args:
        mod (torch.nn.Module): The root module to search in
        qualified_name (str): The qualified name of the attribute (e.g. "submod.param")

    Returns:
        torch.Tensor | None: The tensor attribute if found and is a tensor, None otherwise
    """
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


def set_attr_reference(mod: torch.nn.Module, qualified_name: str, value: torch.Tensor) -> None:
    """Set a tensor attribute on a module by its qualified name.

    Args:
        mod (torch.nn.Module): The root module to set the attribute on
        qualified_name (str): The qualified name of the attribute (e.g. "submod.param")
        value (torch.Tensor): The tensor value to set
    """
    module_path, _, name = qualified_name.rpartition(".")

    try:
        submod: torch.nn.Module = mod.get_submodule(module_path)
    except AttributeError:
        logger.warning(f"Failed to fetch module {module_path}!")
        return

    if isinstance(value, torch.nn.Parameter):
        submod.register_parameter(name, value)
    else:
        submod.register_buffer(name, value)
