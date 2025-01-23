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

from abc import ABC, abstractmethod

from torch.fx import Node
from typing_extensions import Self

from ...types import StrictlyTyped
from ..nodes.node_specialization import NodeSpecialization


class Subgraph(StrictlyTyped, ABC):
    @classmethod
    @abstractmethod
    def configure_from(cls, node: Node) -> Self | None:
        ...

    @property
    def nodes(self) -> list[Node]:
        all_nodes: list[Node] = []
        for name in self.model_dump():
            if isinstance((attr := getattr(self, name, None)), NodeSpecialization):
                all_nodes.append(attr.node)
            elif isinstance(attr, Node):
                all_nodes.append(attr)
            elif isinstance(attr, Subgraph):
                all_nodes.extend(attr.nodes)
        return all_nodes

    def __hash__(self) -> int:
        assert (
            node_hashes := [hash(node) for node in self.nodes]
        ), f"{type(self).__name__} does not have specialized node attributes."
        return sum(node_hashes)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Subgraph) and len(self.nodes) == len(other.nodes):
            return all(x is y for x, y in zip(self.nodes, other.nodes))
        return False
