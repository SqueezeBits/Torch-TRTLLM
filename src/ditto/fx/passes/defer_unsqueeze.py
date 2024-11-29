# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false, reportArgumentType=false

from inspect import isclass
from types import UnionType
from typing import Any, Generic, TypeVar, get_args

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_manager import PassManager, PassResult

from ...constants import FX_TRANSFORM_MAXIMUM_ITERATION
from ...types import Number
from ..nodes import (
    BinaryElementwise,
    CallFunction,
    Embedding,
    IndexSelectNode,
    NodeSpecialization,
    Reduction,
    SymSizeInt,
    UnaryElementwise,
    Unsqueeze,
)
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .graph_pass import GraphOptimizationPass
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, ReplaceAmongInputs


class DeferUnsqueeze(GraphOptimizationPass):
    """Defer the unsqueeze ops as much as possible."""

    def __init__(self, *, depth: int = 0) -> None:
        super().__init__(depth=depth)
        self.pass_manager = PassManager(
            passes=[
                SwapUnsqueezeWithEmbeddingNode(depth=depth + 1),
                SwapUnsqueezeWithBinaryElementwiseNode(depth=depth + 1),
                SwapUnsqueezeWithIndexSelectNode(depth=depth + 1),
                SwapUnsqueezeWithReductionIntListNode(depth=depth + 1),
                SwapUnsqueezeWithUnaryElementwiseNode(depth=depth + 1),
                SwapUnsqueezeWithSymSizeInt(depth=depth + 1),
            ],
            steps=FX_TRANSFORM_MAXIMUM_ITERATION,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        return self.pass_manager(graph_module)


SomeATenOpNode = TypeVar("SomeATenOpNode", bound=CallFunction)


class EarlyExit(Exception):  # noqa: N818
    def __init__(self, pass_results: dict[Node, NodewisePassResult], *args: object) -> None:
        super().__init__(*args)
        self.replacements = pass_results


class SwapUnsqueezeWith(Generic[SomeATenOpNode], NodewiseOptimizationPass):
    """Swap the unsqueeze followed by a child node."""

    @classmethod
    def parent_keys(cls) -> tuple[str, ...]:
        return ("this",)

    @classmethod
    def verify_child(cls, node: Node) -> SomeATenOpNode | None:
        # pylint: disable-next=no-member
        type_arg = get_args(cls.__orig_bases__[0])[0]  # type: ignore[attr-defined]  # noqa: N806
        if isinstance(type_arg, UnionType):
            type_args = get_args(type_arg)
        else:
            type_args = (type_arg,)
        for type_arg in type_args:
            assert isclass(type_arg) and issubclass(
                type_arg, NodeSpecialization
            ), f"Wrong specialization of {cls.__name__} with type parameter {type_arg}"
            if n := type_arg.specialize_from(node):
                return n  # type: ignore[return-value]
        return None

    @classmethod
    def verify_parents(cls, child: SomeATenOpNode) -> dict[str, Unsqueeze]:
        parents: dict[str, Unsqueeze] = {}
        for input_name in cls.parent_keys():
            assert hasattr(child, input_name), (
                f"No such attribute found in class {type(child).__name__}: {input_name}. "
                f"Fix the implementation of the method {cls.__name__}.parent_keys"
            )
            if not (
                isinstance(the_input := getattr(child, input_name), Node)
                and (unsqueeze := Unsqueeze.specialize_from(the_input))
            ):
                continue
            parents[input_name] = unsqueeze
        return parents

    @classmethod
    def get_hotfix_and_unsqueeze_dim(
        cls,
        parents: dict[str, Unsqueeze],
        # pylint: disable-next=unused-argument
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
        hotfix: dict[str, Any] = {name: parents[name].this for name in cls.parent_keys() if name in parents}
        first_unsqueeze = [*parents.values()][0]
        return hotfix, first_unsqueeze.dim

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (child := self.verify_child(node))
            and (unsqueezes := self.verify_parents(child))
            and callable(child_target := child.target)
        ):
            return {}

        try:
            hotfix, unsqueeze_dim = self.get_hotfix_and_unsqueeze_dim(unsqueezes, child)
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
        return {child.node: ReplaceAllUses(by=new_unsqueeze)}


class SwapUnsqueezeWithEmbeddingNode(SwapUnsqueezeWith[Embedding]):
    """Swap the unsqueeze followed by embedding.default node."""

    @classmethod
    def parent_keys(cls) -> tuple[str, ...]:
        return ("indices",)


class SwapUnsqueezeWithIndexSelectNode(SwapUnsqueezeWith[IndexSelectNode]):
    """Swap the unsqueeze followed by index_select.default node."""

    @classmethod
    def get_hotfix_and_unsqueeze_dim(
        cls,
        parents: dict[str, Unsqueeze],
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


class SwapUnsqueezeWithReductionIntListNode(SwapUnsqueezeWith[Reduction]):
    """Swap the unsqueeze followed by a reduction node with dim param."""

    @classmethod
    def get_hotfix_and_unsqueeze_dim(
        cls,
        parents: dict[str, Unsqueeze],
        child: Reduction,
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
                raise EarlyExit({child.node: ReplaceAllUses(by=first_unsqueeze.node)})
            raise EarlyExit({child.node: ReplaceAllUses(by=first_unsqueeze.this)})

        if unsqueeze_dim < child_dim:
            child_dim -= 1
        elif not child.keepdim:
            unsqueeze_dim -= 1
        hotfix["dim"] = [child_dim]
        return hotfix, unsqueeze_dim


class SwapUnsqueezeWithBinaryElementwiseNode(SwapUnsqueezeWith[BinaryElementwise]):
    """Swap the unsqueeze followed by a binary elementwise node possibly with alpha."""

    @classmethod
    def parent_keys(cls) -> tuple[str, ...]:
        return ("this", "other")

    @classmethod
    def verify_parents(cls, child: BinaryElementwise) -> dict[str, Unsqueeze]:
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
            if isinstance(the_other_input := getattr(child, "this" if "other" in unsqueezes else "other"), Number):
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


class SwapUnsqueezeWithUnaryElementwiseNode(SwapUnsqueezeWith[UnaryElementwise]):
    """Swap the unsqueeze followed by a unary elementwise node."""


class SwapUnsqueezeWithSymSizeInt(SwapUnsqueezeWith[SymSizeInt]):
    """Swap the unsqueeze followed by a sym_size.int node."""

    @classmethod
    def get_hotfix_and_unsqueeze_dim(
        cls,
        parents: dict[str, Unsqueeze],
        child: SymSizeInt,
    ) -> tuple[dict[str, Any], int]:
        first_unsqueeze = [*parents.values()][0]
        if (
            (child_dim := child.nonnegative_dim) is None
            or (unsqueeze_dim := first_unsqueeze.nonnegative_dim) is None
            # TODO: when `unsqueeze_dim == child_dim` the child node can be replaced by constant `1`
            or unsqueeze_dim == child_dim
        ):
            raise EarlyExit({})

        if unsqueeze_dim < child_dim:
            child.node.args = (child.node.args[0], child_dim - 1)
            child.node.kwargs = {}
        raise EarlyExit({child.node: ReplaceAmongInputs(occurences_of=first_unsqueeze.node, by=first_unsqueeze.this)})


def get_squeezed_shape(shape: torch.Size, dim: int) -> torch.Size:
    assert 0 <= dim < len(shape)
    return torch.Size((*shape[:dim], *shape[dim + 1 :]))


def append_stack_trace(node: Node, existing_node: Node, pass_name: str) -> None:
    if (existing_stack_trace := existing_node.stack_trace) is not None:
        node.stack_trace = f"{existing_stack_trace}, pass: moved by {pass_name}"
