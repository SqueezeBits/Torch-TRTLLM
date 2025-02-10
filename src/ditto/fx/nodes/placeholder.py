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

from typing import TYPE_CHECKING, Literal

import torch
from torch.fx import Graph, Node
from typing_extensions import Self

from ..utils import get_fake_mode
from .node_specialization import FinalSpecialization

if TYPE_CHECKING:
    from ...arguments import TensorTypeHint


class Placeholder(FinalSpecialization):
    """The specialization for placeholder nodes."""

    @classmethod  # pylint: disable-next=arguments-differ
    def create(
        cls,
        graph: Graph,
        name: str,
        hint: "TensorTypeHint",
    ) -> Self:
        x = cls._specialize_from(graph.placeholder(name))
        if fake_mode := get_fake_mode(graph):
            with fake_mode:
                x.output = torch.empty(hint.symbolic_shape, dtype=hint.dtype)
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
