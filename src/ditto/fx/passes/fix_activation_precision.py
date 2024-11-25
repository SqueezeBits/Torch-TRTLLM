from inspect import isclass
from typing import Generic, TypeVar, get_args

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassManager, PassResult

from ...config import FX_TRANSFORM_MAXIMUM_ITERATION
from ..nodes import Activation
from ..subgraphs import ActivationSubgraph, Silu
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import GraphOptimizationPass, NodeWiseOptimizationPass


class FixActivationPrecision(GraphOptimizationPass):
    """Fix the activation layer precisions."""

    def __init__(self, *, dtype: torch.dtype = torch.float16, depth: int = 0) -> None:
        super().__init__(depth=depth)
        self.pass_manager = PassManager(
            passes=[
                FixSiluPrecision(to=dtype, depth=depth + 1),
                FixNodePrecision(to=dtype, depth=depth + 1),
            ],
            steps=FX_TRANSFORM_MAXIMUM_ITERATION,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        return self.pass_manager(graph_module)


class FixPrecision(NodeWiseOptimizationPass):
    def __init__(self, *, to: torch.dtype = torch.float16, depth: int = 0) -> None:
        super().__init__(depth=depth)
        self.dtype = to


# pylint: disable-next=invalid-name
SubgraphType = TypeVar("SubgraphType", bound=ActivationSubgraph)


class FixSubgraphPrecision(Generic[SubgraphType], FixPrecision):
    @property
    def subgraph_class(self) -> type[ActivationSubgraph]:
        cls = type(self)
        # pylint: disable-next=no-member
        type_arg = get_args(cls.__orig_bases__[0])[0]  # type: ignore[attr-defined]  # noqa: N806
        assert isclass(type_arg) and issubclass(
            type_arg, ActivationSubgraph
        ), f"Wrong specialization of {cls.__name__} with type parameter {type_arg}"
        return type_arg

    def rewrite(self, node: Node) -> dict[Node, Node]:
        if not (
            (subgraph := self.subgraph_class.configure_from(node))
            and (input_meta := get_tensor_metadata(subgraph.input))
            and (output_meta := get_tensor_metadata(subgraph.output))
            and input_meta.dtype == output_meta.dtype
            and input_meta.dtype != self.dtype
        ):
            return {}
        insert_cast(subgraph.input, self.dtype)
        insert_cast(subgraph.output, self.dtype)
        for n in subgraph.nodes:
            if not (meta := get_tensor_metadata(n)):
                continue
            populate_tensor_metadata(n, meta, dtype=self.dtype)
        return {subgraph.input: subgraph.input}


class FixSiluPrecision(FixSubgraphPrecision[Silu]):
    ...


class FixNodePrecision(FixPrecision):
    def rewrite(self, node: Node) -> dict[Node, Node]:
        if not (
            (activation := Activation.specialize_from(node))
            and (input_meta := get_tensor_metadata(activation.this))
            and (output_meta := get_tensor_metadata(activation.node))
            and input_meta.dtype == output_meta.dtype
            and input_meta.dtype != self.dtype
        ):
            return {}
        insert_cast(activation.this, self.dtype)
        insert_cast(activation.node, self.dtype)
        populate_tensor_metadata(node, output_meta, dtype=self.dtype)
        return {node: node}


def insert_cast(x: Node, dtype: torch.dtype) -> None:
    with x.graph.inserting_after(x):
        input_cast = x.graph.call_function(torch.ops.aten._to_copy.default, (x,), {"dtype": dtype})
    if x.stack_trace:
        input_cast.stack_trace = f"{x.stack_trace}, pass: inserted by FixActivationPrecision"
    if meta := get_tensor_metadata(x):
        populate_tensor_metadata(input_cast, meta, dtype=dtype)
    for user in [*x.users]:
        if user == input_cast:
            continue
        user.replace_input_with(x, input_cast)
