# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false, reportArgumentType=false

import warnings
from functools import cached_property
from inspect import isclass
from types import UnionType
from typing import Any, Generic, TypeVar, get_args

import torch
from pydantic.warnings import GenericBeforeBaseModelWarning
from torch.fx import GraphModule, Node

from ...constants import FX_TRANSFORM_MAXIMUM_ITERATION
from ...types import Number
from ..nodes import (
    ATenOp,
    BinaryElementwise,
    Embedding,
    GetAttr,
    IndexSelect,
    NodeSpecialization,
    Reduction,
    Slice,
    SymSizeInt,
    UnaryElementwise,
    Unsqueeze,
)
from ..utils import get_tensor_metadata
from .infra import (
    GraphOptimizationPass,
    NodewiseOptimizationPass,
    NodewisePassResult,
    PassManager,
    PassResult,
    ReplaceAllUses,
    ReplaceAmongInputs,
    inject_stack_trace_from,
)


class DeferUnsqueeze(GraphOptimizationPass):
    """Defer the unsqueeze ops as much as possible."""

    register_create_node_hook: bool = False

    @cached_property
    def pass_manager(self) -> PassManager:
        return PassManager(
            passes=[
                SwapUnsqueezeWithEmbedding(depth=self.depth + 1),
                SwapUnsqueezeWithBinaryElementwise(depth=self.depth + 1),
                SwapUnsqueezeWithIndexSelectOrSlice(depth=self.depth + 1),
                SwapUnsqueezeWithReductionIntList(depth=self.depth + 1),
                SwapUnsqueezeWithUnaryElementwise(depth=self.depth + 1),
                SwapUnsqueezeWithSymSizeInt(depth=self.depth + 1),
            ],
            steps=FX_TRANSFORM_MAXIMUM_ITERATION,
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        return self.pass_manager(graph_module)


SomeATenOp = TypeVar("SomeATenOp", bound=ATenOp)


class EarlyExit(Exception):  # noqa: N818
    def __init__(self, pass_results: dict[Node, NodewisePassResult], *args: object) -> None:
        super().__init__(*args)
        self.replacements = pass_results


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=GenericBeforeBaseModelWarning)

    class SwapUnsqueezeWith(Generic[SomeATenOp], NodewiseOptimizationPass):
        """Swap the unsqueeze followed by a child node."""

        @classmethod
        def parent_keys(cls) -> tuple[str, ...]:
            return ("this",)

        @classmethod
        def verify_child(cls, node: Node) -> SomeATenOp | None:
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
        def verify_parents(cls, child: SomeATenOp) -> dict[str, Unsqueeze]:
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
            child: SomeATenOp,
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
            if not ((child := self.verify_child(node)) and (unsqueezes := self.verify_parents(child))):
                return {}

            try:
                hotfix, unsqueeze_dim = self.get_hotfix_and_unsqueeze_dim(unsqueezes, child)
            except EarlyExit as e:
                return e.replacements

            graph = node.graph
            with graph.inserting_before(child.node):
                new_child = graph.call_function(child.target, *child.args_kwargs(**hotfix))
                inject_stack_trace_from(child, to=new_child)
                new_unsqueeze = Unsqueeze.create(graph, new_child, unsqueeze_dim)
            return {child.node: ReplaceAllUses(by=new_unsqueeze.node)}

    class SwapUnsqueezeWithEmbedding(SwapUnsqueezeWith[Embedding]):
        """Swap the unsqueeze followed by embedding.default node."""

        @classmethod
        def parent_keys(cls) -> tuple[str, ...]:
            return ("indices",)

    class SwapUnsqueezeWithIndexSelectOrSlice(SwapUnsqueezeWith[IndexSelect | Slice]):
        """Swap the unsqueeze followed by index_select.default or slice.Tensor node."""

        @classmethod
        def get_hotfix_and_unsqueeze_dim(
            cls,
            parents: dict[str, Unsqueeze],
            child: IndexSelect | Slice,
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
                child (IndexSelectNode | Slice): the child node of the unsqueezes

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
            if (child_dim := child.nonnegative_dim) is None or (
                unsqueeze_dim := first_unsqueeze.nonnegative_dim
            ) is None:
                raise EarlyExit({})

            if unsqueeze_dim == child_dim:
                raise EarlyExit({})

            if unsqueeze_dim < child_dim:
                child_dim -= 1
            hotfix["dim"] = child_dim
            return hotfix, unsqueeze_dim

    class SwapUnsqueezeWithReductionIntList(SwapUnsqueezeWith[Reduction]):
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
            if (child_dim := child.nonnegative_dim) is None or (
                unsqueeze_dim := first_unsqueeze.nonnegative_dim
            ) is None:
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

    class SwapUnsqueezeWithBinaryElementwise(SwapUnsqueezeWith[BinaryElementwise]):
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
                    and (get_attr := GetAttr.specialize_from(the_other_input))
                    and (output := get_tensor_metadata(child.node))
                ):
                    return {}
                # The case iii) the constant pointed by get_attr must be broadcasted
                # to the parent unsqueeze's output shape.
                try:
                    param_shape = get_attr.parameter.shape
                    _, new_unsqueeze_dim = cls.get_hotfix_and_unsqueeze_dim(unsqueezes, child)
                    unsqueezed_shape = get_squeezed_shape(output.shape, new_unsqueeze_dim)
                    if unsqueezed_shape != torch.broadcast_shapes(param_shape, unsqueezed_shape):
                        return {}
                except (AttributeError, EarlyExit, RuntimeError):
                    return {}
            return unsqueezes

    class SwapUnsqueezeWithUnaryElementwise(SwapUnsqueezeWith[UnaryElementwise]):
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
            raise EarlyExit(
                {child.node: ReplaceAmongInputs(occurrences_of=first_unsqueeze.node, by=first_unsqueeze.this)}
            )


def get_squeezed_shape(shape: torch.Size, dim: int) -> torch.Size:
    assert 0 <= dim < len(shape)
    return torch.Size((*shape[:dim], *shape[dim + 1 :]))
