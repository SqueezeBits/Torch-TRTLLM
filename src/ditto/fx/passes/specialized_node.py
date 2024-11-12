# pyright: reportAttributeAccessIssue=false, reportReturnType=false
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, overload

import torch
from loguru import logger
from pydantic import Field, ValidationError, model_validator
from pydantic_core import PydanticUndefined
from torch.fx.node import Argument, Node, Target
from typing_extensions import Self

from ...fake_targets import fake_transposed_mm
from ...types import StrictlyTyped
from ...utils import make_axis_nonnegative, make_dim_nonnegative
from ..utils import get_tensor_metadata


class SpecializedNode(StrictlyTyped, ABC):
    """Abstract base class for defining node whose arguments are specialized for a specific op and target(s)."""

    node: Node = Field(exclude=True, frozen=True)

    @classmethod
    @abstractmethod
    def designated_op(cls) -> str:
        ...

    @classmethod
    @abstractmethod
    def possible_targets(cls) -> tuple[Target, ...]:
        ...

    @property
    def op(self) -> str:
        return self.node.op

    @property
    def target(self) -> Target:
        return self.node.target

    @property
    def name(self) -> str:
        return self.node.name

    def args_kwargs(self, **hotfix: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        _args: list[Any] = []
        _kwargs: dict[str, Any] = {}
        append_in_args = True
        data = self.model_dump()
        if hotfix:
            data.update(hotfix)
        for name, field in self.model_fields.items():
            if field.exclude:
                if name == "asterick":
                    append_in_args = False
                continue
            value = data[name]
            if append_in_args:
                _args.append(value)
            elif field.is_required() or value != field.get_default(call_default_factory=True):
                _kwargs[name] = value
        return tuple(_args), _kwargs

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        """Validate if the node is suitable for argument extraction.

        Args:
            node (Node): a node

        Returns:
            bool: `True` if it is suitable, `False` otherwise.
        """
        return node.op == cls.designated_op() and node.target in cls.possible_targets()

    @classmethod
    def specialize_from(cls, node: Node) -> Self | None:
        """Specialize from the given node.

        Args:
            node (Node): a node

        Returns:
            Self | None: the specialized node if succeeded, `None` otherwise.
        """
        if not cls.validate_node(node):
            return None
        try:
            arguments = {
                name: get_argument(
                    node,
                    index - 1,
                    name,
                    default=None if field.default is PydanticUndefined else field.default,
                )
                for index, (name, field) in enumerate(cls.model_fields.items())
                if index > 0  # should skip the `node`
            }
            return cls.model_validate({"node": node, **arguments})
        except ValidationError as e:
            logger.warning(f"Incorrect arguments given to the node {node.format_node()}: {arguments}. ({e})")
            return None


class ATenOpNode(SpecializedNode):
    @property
    def target(self) -> Callable[..., Any]:
        assert callable(op := super().target)
        return op

    @classmethod
    def designated_op(cls) -> str:
        return "call_function"

    @classmethod
    @abstractmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        ...


class GetAttrNode(SpecializedNode):
    @property
    def target(self) -> str:
        assert isinstance(name := super().target, str)
        return name

    @classmethod
    def designated_op(cls) -> str:
        return "call_function"

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        if not (
            super().validate_node(node)
            and (graph_module := node.graph.owning_module)
            and isinstance(name := node.target, str)
        ):
            return False
        try:
            _ = graph_module.get_parameter(name)
            return True
        except AttributeError:
            return False

    @property
    def parameter(self) -> torch.nn.Parameter:
        assert (graph_module := self.node.graph.owning_module) is not None
        return graph_module.get_parameter(self.target)


DefaultValue = TypeVar("DefaultValue")


@overload
def get_argument(
    node: Node,
    index_as_arg: int,
    name_as_kwarg: str,
    default: DefaultValue,
) -> DefaultValue:
    ...


@overload
def get_argument(
    node: Node,
    index_as_arg: int,
    name_as_kwarg: str,
) -> Argument:
    ...


def get_argument(
    node: Node,
    index_as_arg: int,
    name_as_kwarg: str,
    default: DefaultValue | None = None,
) -> DefaultValue | Argument:
    """Get the node argument of the given node.

    Args:
        node (Node): a node
        index_as_arg (int): the index to look up when the node argument is given as a positional argument
        name_as_kwarg (str): the key to look up when the node argument is given as a keyword argument
        default (DefaultValue | None, optional): the default value when the node argument is not explicitly specified.
            Defaults to None.

    Returns:
        DefaultValue | Argument: the node argument if found or its default value.
    """
    return (
        node.kwargs[name_as_kwarg]
        if name_as_kwarg in node.kwargs
        else (node.args[index_as_arg] if len(node.args) > index_as_arg else default)
    )


Asterick = Field(default=None, exclude=True)
SymInt = int | torch.SymInt | Node
Number = int | float | bool


class BinaryElementwiseNode(ATenOpNode):
    x: Node | Number
    y: Node | Number

    @model_validator(mode="after")
    def check_if_one_of_the_inputs_is_node(self) -> Self:
        if not (isinstance(self.x, Node) or isinstance(self.y, Node)):
            raise ValidationError(
                f"Expected one of the inputs of {self.node} to be a `torch.fx.Node` but got x={self.x}, y={self.y}"
            )
        return self

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.div.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.pow.Scalar,
            torch.ops.aten.pow.Tensor_Scalar,
            torch.ops.aten.pow.Tensor_Tensor,
        )


class BinaryElementwiseWithAlphaNode(BinaryElementwiseNode):
    asterick: None = Asterick
    alpha: Number = 1

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
        )


class CombineNode(ATenOpNode):
    tensors: list[Node]
    dim: int = 0

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.cat.default,
            torch.ops.aten.stack.default,
        )


class ReductionIntListNode(ATenOpNode):
    x: Node
    dim: list[int] = Field(max_length=1, min_length=1)
    keepdim: bool = False
    asterick: None = Asterick
    dtype: torch.dtype | None = None

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.mean.dim,
            torch.ops.aten.sum.dim_IntList,
        )

    @property
    def input_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.x):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim[0], ndim=ndim)
        return None


class SingleDimensionReshape(ATenOpNode):
    x: Node
    dim: int

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.unsqueeze.default,
        )

    @property
    def input_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.x):
            return len(t.shape)
        return None

    @property
    def output_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None


class UnaryElementwiseNode(ATenOpNode):
    x: Node

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.sqrt.default,
        )


class AddNode(BinaryElementwiseWithAlphaNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.add.Tensor,)


class CatNode(CombineNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.cat.default,)

    @property
    def ndim(self) -> int | None:
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


class CloneNode(ATenOpNode):
    x: Node
    asterick: None = Asterick
    memory_format: torch.memory_format | None = None

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.clone.default,)


class DivNode(BinaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.div.Tensor,)


class EmbeddingNode(ATenOpNode):
    weight: Node
    indices: Node
    padding_idx: SymInt = -1
    scale_grad_by_freq: bool = False
    sparse: bool = False

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.embedding.default,)


class IndexSelectNode(ATenOpNode):
    x: Node
    dim: int
    index: Node

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.index_select.default,)

    @property
    def output_ndim(self) -> int | None:
        if t := get_tensor_metadata(self.node):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


class MeanDimNode(ReductionIntListNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.mean.dim,)


class MMNode(ATenOpNode):
    lhs: Node
    rhs: Node

    @property
    def is_rhs_transposed(self) -> bool:
        return self.target is fake_transposed_mm

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.mm.default,
            fake_transposed_mm,
        )


class MMConstNode(MMNode):
    @classmethod
    def validate_node(cls, node: Node) -> bool:
        if (
            super().validate_node(node)
            and (rhs := node.all_input_nodes[1]).op == "get_attr"
            and isinstance(target := rhs.target, str)
            and (graph_module := rhs.graph.owning_module)
        ):
            try:
                _ = graph_module.get_parameter(target)
                return True
            except AttributeError:
                pass
        return False

    @property
    def weight_name(self) -> str:
        assert isinstance(target := self.rhs.target, str)
        return target

    @property
    def weight(self) -> torch.nn.Parameter:
        assert (graph_module := self.rhs.graph.owning_module)
        return graph_module.get_parameter(self.weight_name)


class MulNode(BinaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.mul.Tensor,)


class PermuteNode(ATenOpNode):
    x: Node
    dims: list[int]

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.permute.default,)

    @property
    def ndim(self) -> int:
        return len(self.dims)


class PowNode(BinaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (
            torch.ops.aten.pow.Scalar,
            torch.ops.aten.pow.Tensor_Scalar,
            torch.ops.aten.pow.Tensor_Tensor,
        )


class ReshapeNode(ATenOpNode):
    x: Node
    shape: list[SymInt]

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.reshape.default,)

    @property
    def target_shape(self) -> torch.Size | None:
        sym_ints: list[int | torch.SymInt] = []
        for s in self.shape:
            if isinstance(s, int | torch.SymInt):
                sym_ints.append(s)
                continue
            if not isinstance(val := s.meta.get("val"), torch.SymInt):
                return None
            sym_ints.append(val)
        return torch.Size(sym_ints)  # type: ignore[arg-type]


class SDPANode(SpecializedNode):
    query: Node
    key: Node
    value: Node
    attn_mask: Node | None = None
    dropout_p: float = 0.0
    is_causal: bool = False
    scale: float | None = None
    enable_gqa: bool = False

    @classmethod
    def designated_op(cls) -> str:
        return "call_function"

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch._C._nn.scaled_dot_product_attention,)

    @property
    def is_eligible_for_gpt_attention_plugin(self) -> bool:
        for name, field in self.model_fields.items():
            if field.is_required():
                continue
            if (value := getattr(self, name)) != (default_value := field.get_default()):
                logger.warning(
                    f"Cannot support the non-default '{name}={value}' provided to `F.scaled_dot_product_attention` "
                    f"(default is {default_value})"
                )
                return False
        return True


class SliceNode(ATenOpNode):
    x: Node
    dim: int = 0
    start: SymInt | None = None
    end: SymInt | None = None
    step: SymInt = 1

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.slice.Tensor,)

    @property
    def dim_size(self) -> int | None:
        if (t := get_tensor_metadata(self.x)) and isinstance(s := t.shape[self.dim], int):
            return s
        return None

    @property
    def ndim(self) -> int | None:
        if t := get_tensor_metadata(self.x):
            return len(t.shape)
        return None

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None

    @property
    def nonnegative_start(self) -> int | None:
        if isinstance(self.start, int) and (dim_size := self.dim_size) is not None:
            return make_axis_nonnegative(self.start, dim_size=dim_size)
        return None

    @property
    def nonnegative_end(self) -> int | None:
        if isinstance(self.end, int) and (dim_size := self.dim_size) is not None:
            return make_axis_nonnegative(self.end, dim_size=dim_size)
        return None


class SqrtNode(UnaryElementwiseNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.sqrt.default,)


class SplitNode(ATenOpNode):
    x: Node
    split_size: list[SymInt] | SymInt
    dim: int = 0

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.split.default, torch.ops.aten.split.sizes)


class StackNode(CombineNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.stack.default,)


class SqueezeDimNode(SingleDimensionReshape):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.squeeze.dim,)

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.input_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None


class SubNode(BinaryElementwiseWithAlphaNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.sub.Tensor,)


class SumDimIntListNode(ReductionIntListNode):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.sum.dim_IntList,)


class ToCopyNode(ATenOpNode):
    x: Node
    asterick: None = Asterick
    dtype: torch.dtype | None = None
    layout: torch.layout | None = None
    device: torch.device | None = None
    pin_memory: bool | None = None
    non_blocking: bool = False
    memory_format: torch.memory_format | None = None

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten._to_copy.default,)


class UnsqueezeNode(SingleDimensionReshape):
    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        return (torch.ops.aten.unsqueeze.default,)

    @property
    def nonnegative_dim(self) -> int | None:
        if (ndim := self.output_ndim) is not None:
            return make_dim_nonnegative(self.dim, ndim=ndim)
        return None
