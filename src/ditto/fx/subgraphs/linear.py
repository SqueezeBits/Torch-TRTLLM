import torch
from torch._subclasses import FakeTensor
from torch.fx import Node
from typing_extensions import Self

from ditto.fx.utils import get_val

from ..nodes import MM, Add, Reshape
from .subgraph import Subgraph


class AddTensor(Add):
    """Add node with both operands being tensors."""

    this: Node
    other: Node


class Linear(Subgraph):
    """A subgraph representing a linear layer.

    This subgraph identifies a pattern of matrix multiplication with an optional bias addition,
    which is equivalent to a linear/dense layer in neural networks.

    The layer performs: output = input @ weight.T + bias

    Attributes:
        mm (MM): The matrix multiplication operation node
        add (AddTensor | None): The bias addition operation node, if present
    """

    mm: MM
    add: AddTensor | None

    @property
    def weight_node(self) -> Node:
        """Get the weight parameter node."""
        return self.mm.other

    @property
    def weight_tensor(self) -> FakeTensor:
        """Get the weight parameter tensor."""
        assert (weight := get_val(self.mm.other, FakeTensor)) is not None
        return weight

    @property
    def bias_node(self) -> Node | None:
        """Get the bias parameter node if present."""
        return self.add.other if self.add is not None else None

    @property
    def bias_tensor(self) -> FakeTensor | None:
        """Get the bias parameter tensor if present."""
        if not (self.add is not None and (bias := get_val(self.add.other, FakeTensor))):
            return None
        return bias

    @property
    def input_node(self) -> Node:
        """Get the input tensor node to the linear layer."""
        return self.mm.this

    @property
    def output_node(self) -> Node:
        """Get the output tensor node, either the bias addition or matrix multiplication result."""
        return self.add.node if self.add is not None else self.mm.node

    @property
    def reshape_in(self) -> Reshape | None:
        """Get the reshape operation before the linear layer if present."""
        return Reshape.specialize_from(self.mm.this)

    @property
    def reshape_out(self) -> Reshape | None:
        """Get the reshape operation after the linear layer if present."""
        if len(users := list(self.output_node.users)) != 1:
            return None
        return Reshape.specialize_from(users[0])

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not ((mm := MM.specialize_from(node)) and (weight := get_val(mm.other, torch.Tensor)) is not None):
            return None

        add = AddTensor.specialize_from(users[0]) if len(users := list(mm.users)) == 1 else None
        if add is not None and not (
            add.this == mm.node
            and (bias := get_val(add.other, torch.Tensor)) is not None
            and bias.shape[-1] == weight.shape[-1]
        ):
            add = None
        return cls(mm=mm, add=add)
