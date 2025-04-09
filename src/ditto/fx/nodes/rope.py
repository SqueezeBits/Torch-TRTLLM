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

from collections.abc import Callable
from typing import Any

from torch.fx import Graph
from torch.fx.node import Argument
from typing_extensions import Self

from ..targets import FAKE_ROPE_TARGETS
from .call_function import FinalCallFunction
from .node_specialization import NodeSpecialization


class Rope(FinalCallFunction):
    """Represents a wrapped Rotary Position Embedding (RoPE) node in the computation graph.

    This class specializes FinalCallFunction for RoPE operations, which apply rotary
    position embeddings. RoPE subgraph is wrapped by this node.
    """

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        """Get the possible target functions that this node can represent.

        Returns:
            tuple[Callable[..., Any], ...]: Tuple of callable targets from FAKE_ROPE_TARGETS
        """
        return tuple(FAKE_ROPE_TARGETS.values())

    @classmethod
    def create_with_target(
        cls,
        graph: Graph,
        target: Callable[..., Any],
        *args: Argument | NodeSpecialization,
        **kwargs: Argument | NodeSpecialization,
    ) -> Self:
        """Create a new RoPE node in the computation graph.

        Args:
            graph (Graph): The computation graph to add the node to
            target (Callable[..., Any]): The RoPE function to call
            *args (Argument | NodeSpecialization): Positional arguments for the RoPE function
            **kwargs (Argument | NodeSpecialization): Keyword arguments for the RoPE function

        Returns:
            Self: A new specialized RoPE node
        """
        assert target in cls.possible_targets()
        args_, kwargs_ = cls.unwrap_specialization(*args, **kwargs)
        node = graph.call_function(target, args_, kwargs_)
        return cls._specialize_from(node)
