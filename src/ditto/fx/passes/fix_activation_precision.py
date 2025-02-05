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

import warnings
from functools import cached_property
from inspect import isclass
from typing import Generic, TypeVar, get_args

import torch
from pydantic.warnings import GenericBeforeBaseModelWarning
from torch.fx import GraphModule, Node

from ...constants import FX_TRANSFORM_MAXIMUM_ITERATION
from ..nodes import Activation, ToCopy
from ..subgraphs import ActivationSubgraph, Silu
from ..utils import get_tensor_metadata
from .infra import (
    GraphOptimizationPass,
    ModifiedInsideThePass,
    NodewiseOptimizationPass,
    NodewisePassResult,
    PassManager,
    PassResult,
)


class FixActivationPrecision(GraphOptimizationPass):
    """Fix the activation layer precisions.

    Attributes:
        dtype (torch.dtype): The dtype to fix the activation layer precisions to.
    """

    dtype: torch.dtype

    @cached_property
    def pass_manager(self) -> PassManager:
        """The pass manager for the fix activation precision passes.

        Returns:
            PassManager: The pass manager that contains the fix activation precision passes.
        """
        return PassManager(
            passes=[
                FixSiluPrecision(dtype=self.dtype, depth=self.depth + 1),
                FixActivationNodePrecision(dtype=self.dtype, depth=self.depth + 1),
            ],
            steps=FX_TRANSFORM_MAXIMUM_ITERATION,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        return self.pass_manager(graph_module)


class FixPrecision(NodewiseOptimizationPass):
    """Fix the precision of the activation layer.

    Attributes:
        dtype (torch.dtype): The dtype to fix the activation layer precisions to.
    """

    dtype: torch.dtype


# pylint: disable-next=invalid-name
SubgraphType = TypeVar("SubgraphType", bound=ActivationSubgraph)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=GenericBeforeBaseModelWarning)

    class FixActivationSubgraphPrecision(Generic[SubgraphType], FixPrecision):
        """Fix the precision of the activation subgraph."""

        @property
        def subgraph_class(self) -> type[ActivationSubgraph]:
            """The class of the activation subgraph."""
            cls = type(self)
            # pylint: disable-next=no-member
            type_arg = get_args(cls.__orig_bases__[0])[0]  # noqa: N806
            assert isclass(type_arg) and issubclass(
                type_arg, ActivationSubgraph
            ), f"Wrong specialization of {cls.__name__} with type parameter {type_arg}"
            return type_arg

        def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
            if not (
                (subgraph := self.subgraph_class.configure_from(node))
                and (input_meta := get_tensor_metadata(subgraph.input))
                and (output_meta := get_tensor_metadata(subgraph.output))
                and input_meta.dtype == output_meta.dtype
                and input_meta.dtype != self.dtype
            ):
                return {}
            insert_cast_after(subgraph.input, self.dtype)
            insert_cast_after(subgraph.output, self.dtype)
            return {node: ModifiedInsideThePass()}

    class FixSiluPrecision(FixActivationSubgraphPrecision[Silu]):
        """Fix the precision of the silu subgraph."""


class FixActivationNodePrecision(FixPrecision):
    """Fix the precision of the activation node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (activation := Activation.specialize_from(node))
            and (input_meta := get_tensor_metadata(activation.this))
            and (output_meta := get_tensor_metadata(activation.node))
            and input_meta.dtype == output_meta.dtype
            and input_meta.dtype != self.dtype
        ):
            return {}
        insert_cast_after(activation.this, self.dtype)
        insert_cast_after(activation.node, self.dtype)
        return {node: ModifiedInsideThePass()}


def insert_cast_after(x: Node, dtype: torch.dtype) -> None:
    """Insert a cast after the node.

    Args:
        x (Node): The node to insert the cast after.
        dtype (torch.dtype): The dtype to cast to.
    """
    with x.graph.inserting_after(x):
        input_cast = ToCopy.create(x.graph, x, dtype=dtype).node
    for user in [*x.users]:
        if user == input_cast:
            continue
        user.replace_input_with(x, input_cast)
