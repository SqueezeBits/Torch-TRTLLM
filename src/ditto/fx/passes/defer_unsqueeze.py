# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false, reportArgumentType=false

from inspect import isclass
from types import UnionType
from typing import Any, Generic, TypeVar, get_args

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassManager, PassResult

from ...config import FX_TRANSFORM_MAXIMUM_ITERATION
from ...types import Number
from ..nodes import (
    BinaryElementwiseNode,
    CallFunctionNode,
    CloneNode,
    EmbeddingNode,
    IndexSelectNode,
    ReductionIntListNode,
    SpecializedNode,
    ToCopyNode,
    UnaryElementwiseNode,
    UnsqueezeNode,
)
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .graph_pass import GraphOptimizationPass
from .node_wise_pass import NodeWiseOptimizationPass


class DeferUnsqueeze(GraphOptimizationPass):
    """Defer the unsqueeze ops as much as possible."""

    def __init__(self, depth: int = 0) -> None:
        super().__init__(depth)
        self.pass_manager = PassManager(
            passes=[
                SwapUnsqueezeWithEmbeddingNode(depth + 1),
                SwapUnsqueezeWithBinaryElementwiseNode(depth + 1),
                SwapUnsqueezeWithIndexSelectNode(depth + 1),
                SwapUnsqueezeWithReductionIntListNode(depth + 1),
                SwapUnsqueezeWithUnaryElementwiseNode(depth + 1),
            ],
            steps=FX_TRANSFORM_MAXIMUM_ITERATION,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        return self.pass_manager(graph_module)


SomeATenOpNode = TypeVar("SomeATenOpNode", bound=CallFunctionNode)


class EarlyExit(Exception):  # noqa: N818
    def __init__(self, replacements: dict[Node, Node], *args: object) -> None:
        super().__init__(*args)
        self.replacements = replacements


class SwapUnsqueezeWith(Generic[SomeATenOpNode], NodeWiseOptimizationPass):
    """Swap the unsqueeze followed by a child node."""

    @classmethod
    def parent_keys(cls) -> tuple[str, ...]:
        return ("x",)

    @classmethod
    def verify_child(cls, node: Node) -> SomeATenOpNode | None:
        type_arg = get_args(cls.__orig_bases__[0])[0]  # type: ignore[attr-defined]  # noqa: N806
        if isinstance(type_arg, UnionType):
            type_args = get_args(type_arg)
        else:
            type_args = (type_arg,)
        for type_arg in type_args:
            assert isclass(type_arg) and issubclass(
                type_arg, SpecializedNode
            ), f"Wrong specialization of {cls.__name__} with type parameter {type_arg}"
            if n := type_arg.specialize_from(node):
                return n  # type: ignore[return-value]
        return None

    @classmethod
    def verify_parents(cls, child: SomeATenOpNode) -> dict[str, UnsqueezeNode]:
        parents: dict[str, UnsqueezeNode] = {}
        for input_name in cls.parent_keys():
            assert hasattr(child, input_name), (
                f"No such attribute found in class {type(child).__name__}. "
                f"Fix the implementation of the method {cls.__name__}.parent_keys"
            )
            if not (
                isinstance(the_input := getattr(child, input_name), Node)
                and (unsqueeze := UnsqueezeNode.specialize_from(the_input))
            ):
                continue
            parents[input_name] = unsqueeze
        return parents

    @classmethod
    def get_hotfix_and_unsqueeze_dim(
        cls,
        parents: dict[str, UnsqueezeNode],
        child: SomeATenOpNode,
    ) -> tuple[dict[str, Any], int]:
        """Get the hotfix for the new child node and the unsqueeze dimension for the new unsqueeze node.

        Args:
            parents (dict[str, UnsqueezeArguments]): the parent unsqueezes
            child (ATenOpArgumentsType): the child node of the unsqueezes

        Raises:
            EarlyExit:
                i) when no parents are available
                ii) when there are more than one parents with different `dim` or output dimension size

        Returns:
            tuple[dict[str, Any], int]: the hotfix for the new child node and the unsqueeze dimension for the new
                unsqueeze node
        """
        if not parents:
            raise EarlyExit({})
        hotfix: dict[str, Any] = {name: parents[name].x for name in cls.parent_keys() if name in parents}
        first_unsqueeze = [*parents.values()][0]
        return hotfix, first_unsqueeze.dim

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (child := cls.verify_child(node))
            and (unsqueezes := cls.verify_parents(child))
            and callable(child_target := child.target)
        ):
            return {}

        try:
            hotfix, unsqueeze_dim = cls.get_hotfix_and_unsqueeze_dim(unsqueezes, child)
        except EarlyExit as e:
            return e.replacements
        first_unsqueeze = [*unsqueezes.values()][0]

        graph = node.graph
        original_output = get_tensor_metadata(child.node)
        with graph.inserting_before(child.node):
            new_child = graph.call_function(child_target, *child.args_kwargs(**hotfix))
            append_stack_trace(new_child, child.node, DeferUnsqueeze.__name__)
            if original_output:
                _ = populate_tensor_metadata(
                    new_child, original_output, shape=get_squeezed_shape(original_output.shape, unsqueeze_dim)
                )
            new_unsqueeze = graph.call_function(torch.ops.aten.unsqueeze.default, (new_child, unsqueeze_dim))
            append_stack_trace(new_unsqueeze, first_unsqueeze.node, DeferUnsqueeze.__name__)
            if original_output:
                _ = populate_tensor_metadata(new_unsqueeze, original_output)
        return {child.node: new_unsqueeze}


class SwapUnsqueezeWithEmbeddingNode(SwapUnsqueezeWith[EmbeddingNode]):
    """Swap the unsqueeze followed by embedding.default node."""

    @classmethod
    def parent_keys(cls) -> tuple[str, ...]:
        return ("indices",)


class SwapUnsqueezeWithIndexSelectNode(SwapUnsqueezeWith[IndexSelectNode]):
    """Swap the unsqueeze followed by index_select.default node."""

    @classmethod
    def get_hotfix_and_unsqueeze_dim(
        cls,
        parents: dict[str, UnsqueezeNode],
        child: IndexSelectNode,
    ) -> tuple[dict[str, Any], int]:
        """Further handle axes while swapping an unsqueeze and an index_select node.

        For example:
        ```python
        x = torch.randn(2, 3, 4, 5, 6)
        i = torch.randint(0, 4, (10,))

        # case 1) when unsqueeze.dim == index_select.dim
        # the unsqueeze and index_select cannot be swapped, and it's rather
        # an obscure case as the index tensor must be filled with zeros
        x.unsqueeze(2).index_select(2, torch.zeros_like(i)).shape  # torch.Size([2, 3, 10, 4, 5, 6])

        # case 2) when unsqueeze.dim < mean.dim
        torch.all(x.unsqueeze(2).index_select(4, i) == x.index_select(3, i).unsqueeze(2))  # True

        # case 3) when unsqueeze.dim > mean.dim
        torch.all(x.unsqueeze(4).index_select(2, i) == x.index_select(2, i).unsqueeze(4))  # True
        ```

        Args:
            parents (dict[str, UnsqueezeArguments]): the parent unsqueezes
            child (IndexSelectNode): the child node of the unsqueezes

        Raises:
            EarlyExit:
                i) when no tensor metadata found in one of child or the first parent
                ii) when unsqueeze dimension is the same as the index_select dimension

        Returns:
            tuple[dict[str, Any], int]: the hotfix for the new child node and the unsqueeze dimension for the new
                unsqueeze node
        """
        hotfix, _ = super().get_hotfix_and_unsqueeze_dim(parents, child)
        first_unsqueeze = [*parents.values()][0]
        if (child_dim := child.nonnegative_dim) is None or (unsqueeze_dim := first_unsqueeze.nonnegative_dim) is None:
            raise EarlyExit({})

        if unsqueeze_dim == child_dim:
            raise EarlyExit({})

        if unsqueeze_dim < child_dim:
            child_dim -= 1
        hotfix["dim"] = child_dim
        return hotfix, unsqueeze_dim


class SwapUnsqueezeWithReductionIntListNode(SwapUnsqueezeWith[ReductionIntListNode]):
    """Swap the unsqueeze followed by a reduction node with dim param."""

    @classmethod
    def get_hotfix_and_unsqueeze_dim(
        cls,
        parents: dict[str, UnsqueezeNode],
        child: ReductionIntListNode,
    ) -> tuple[dict[str, Any], int]:
        """Further handle axes while swapping an unsqueeze and a reduction node.

        For example, if you want to swap unsqueeze and mean:
        ```python
        x = torch.randn(2, 3, 4, 5, 6)

        # case 1) when unsqueeze.dim == mean.dim
        torch.all(x.unsqueeze(2).mean(dim=2) == x)  # True
        torch.all(x.unsqueeze(2).mean(dim=2, keepdim=True) == x.unsqueeze(2))  # True

        # case 2) when unsqueeze.dim < mean.dim
        torch.all(x.unsqueeze(1).mean(dim=2) == x.mean(dim=1).unsqueeze(1))  # True
        torch.all(x.unsqueeze(1).mean(dim=2, keepdim=True) == x.mean(dim=1, keepdim=True).unsqueeze(1))  # True

        # case 3) when unsqueeze.dim > mean.dim
        torch.all(x.unsqueeze(3).mean(dim=2) == x.mean(dim=2).unsqueeze(2))  # True
        torch.all(x.unsqueeze(3).mean(dim=2, keepdim=True) == x.mean(dim=2, keepdim=True).unsqueeze(3))  # True
        ```

        Args:
            parents (dict[str, UnsqueezeArguments]): the parent unsqueezes
            child (ReductionIntListArguments): the child node of the unsqueezes

        Raises:
            EarlyExit:
                i) when no tensor metadata found in one of child or the first parent
                ii) when unsqueeze dimension is the same as the reduction dimension

        Returns:
            tuple[dict[str, Any], int]: the hotfix for the new child node and the unsqueeze dimension for the new
                unsqueeze node
        """
        hotfix, _ = super().get_hotfix_and_unsqueeze_dim(parents, child)
        first_unsqueeze = [*parents.values()][0]
        if (child_dim := child.nonnegative_dim) is None or (unsqueeze_dim := first_unsqueeze.nonnegative_dim) is None:
            raise EarlyExit({})

        if unsqueeze_dim == child_dim:
            if child.keepdim:
                raise EarlyExit({child.node: first_unsqueeze.node})
            raise EarlyExit({child.node: first_unsqueeze.x})

        if unsqueeze_dim < child_dim:
            child_dim -= 1
        elif not child.keepdim:
            unsqueeze_dim -= 1
        hotfix["dim"] = [child_dim]
        return hotfix, unsqueeze_dim


class SwapUnsqueezeWithBinaryElementwiseNode(SwapUnsqueezeWith[BinaryElementwiseNode]):
    """Swap the unsqueeze followed by a binary elementwise node possibly with alpha."""

    @classmethod
    def parent_keys(cls) -> tuple[str, ...]:
        return ("x", "y")

    @classmethod
    def verify_parents(cls, child: BinaryElementwiseNode) -> dict[str, UnsqueezeNode]:
        unsqueezes = super().verify_parents(child)
        # for a binary elementwise child, the pass should run only if the inputs must be one of the form:
        # i) (unsqueeze, unsqueeze_1)
        # ii) (unsqueeze, scalar) or (scalar, unsqueeze)
        # iii) (unsqueeze, get_attr) or (get_attr, unsqueeze),
        if len(unsqueezes) == 2:
            # The case i)
            # If there are unsqueeze nodes on both sides, we need several assumptions as follows.
            # Assumption 1: all unsqueeze nodes share the same output dimension sizes
            ndims = {unsqueeze.output_ndim for unsqueeze in unsqueezes.values()}
            if None in ndims or len(ndims) > 1:
                return {}
            # Assumption 2: all unsqueeze nodes share the same nonnegative dim values
            dims = {unsqueeze.nonnegative_dim for unsqueeze in unsqueezes.values()}
            if None in dims or len(dims) > 1:
                return {}
        elif len(unsqueezes) == 1:
            # The case ii)
            if isinstance(the_other_input := getattr(child, "x" if "y" in unsqueezes else "y"), Number):
                return unsqueezes
            if not (
                isinstance(the_other_input, Node)
                and the_other_input.op == "get_attr"
                and isinstance(param_name := the_other_input.target, str)
                and (output := get_tensor_metadata(child.node))
                and (graph_module := the_other_input.graph.owning_module)
            ):
                return {}
            # The case iii) the constant pointed by get_attr must have a broadcastable shape
            try:
                param = graph_module.get_parameter(param_name)
                _, new_unsqueeze_dim = cls.get_hotfix_and_unsqueeze_dim(unsqueezes, child)
                unsqueezed_shape = get_squeezed_shape(output.shape, new_unsqueeze_dim)
                if unsqueezed_shape != torch.broadcast_shapes(param.shape, unsqueezed_shape):
                    return {}
            except (AttributeError, EarlyExit, RuntimeError):
                return {}
        return unsqueezes


class SwapUnsqueezeWithUnaryElementwiseNode(SwapUnsqueezeWith[UnaryElementwiseNode | ToCopyNode | CloneNode]):
    """Swap the unsqueeze followed by a unary elementwise node."""


def get_squeezed_shape(shape: torch.Size, dim: int) -> torch.Size:
    assert 0 <= dim < len(shape)
    return torch.Size((*shape[:dim], *shape[dim + 1 :]))


def append_stack_trace(node: Node, existing_node: Node, pass_name: str) -> None:
    if (existing_stack_trace := existing_node.stack_trace) is not None:
        node.stack_trace = f"{existing_stack_trace}, pass: moved by {pass_name}"
