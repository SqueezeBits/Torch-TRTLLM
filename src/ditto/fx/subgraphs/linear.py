from torch.fx import Node
from typing_extensions import Self

from ..nodes import MM, Add, GetAttr, Reshape
from .subgraph import Subgraph


class WeightMatmul(Subgraph):
    """A subgraph representing a matrix multiplication with a weight parameter.

    This subgraph identifies a pattern of matrix multiplication between an input tensor
    and a weight parameter accessed via GetAttr.

    Attributes:
        mm (MM): The matrix multiplication operation node
        weight (GetAttr): The weight parameter getter node
    """

    mm: MM
    weight: GetAttr

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (mm := MM.specialize_from(node)) and (weight := GetAttr.specialize_from(mm.other)):
            return cls(mm=mm, weight=weight)
        return None


class BiasAdd(Subgraph):
    """A subgraph representing a bias addition operation.

    This subgraph identifies a pattern of adding a 1D bias parameter to a tensor.

    Attributes:
        add (Add): The addition operation node
        bias (GetAttr): The bias parameter getter node
    """

    add: Add
    bias: GetAttr

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (
            (add := Add.specialize_from(node))
            and isinstance(add.other, Node)
            and (bias := GetAttr.specialize_from(add.other))
            and bias.parameter.ndim == 1
        ):
            return cls(add=add, bias=bias)
        return None


class Linear(Subgraph):
    """A subgraph representing a linear layer.

    This subgraph identifies a pattern of matrix multiplication with an optional bias addition,
    which is equivalent to a linear/dense layer in neural networks.

    The layer performs: output = input @ weight.T + bias

    Attributes:
        weight_matmul (WeightMatmul): The weight matrix multiplication subgraph
        bias_add (BiasAdd | None): The bias addition subgraph, if present
    """

    weight_matmul: WeightMatmul
    bias_add: BiasAdd | None

    @property
    def mm(self) -> MM:
        """Get the matrix multiplication node.

        Returns:
            MM: The matrix multiplication operation node
        """
        return self.weight_matmul.mm

    @property
    def weight(self) -> GetAttr:
        """Get the weight parameter node.

        Returns:
            GetAttr: The weight parameter getter node
        """
        return self.weight_matmul.weight

    @property
    def has_bias(self) -> bool:
        """Check if this linear layer has a bias term.

        Returns:
            bool: True if bias is present, False otherwise
        """
        return self.bias_add is not None

    @property
    def add(self) -> Add:
        """Get the bias addition node.

        Returns:
            Add: The bias addition operation node

        Raises:
            AttributeError: If this linear layer has no bias
        """
        if self.bias_add is None:
            raise AttributeError(f"{repr(self)} does not have bias")
        return self.bias_add.add

    @property
    def bias(self) -> GetAttr:
        """Get the bias parameter node.

        Returns:
            GetAttr: The bias parameter getter node

        Raises:
            AttributeError: If this linear layer has no bias
        """
        if self.bias_add is None:
            raise AttributeError(f"{repr(self)} does not have bias")
        return self.bias_add.bias

    @property
    def input(self) -> Node:
        """Get the input node to the linear layer.

        Returns:
            Node: The input tensor node
        """
        return self.mm.this

    @property
    def output(self) -> Node:
        """Get the output node of the linear layer.

        Returns:
            Node: The output tensor node, either the bias addition or matrix multiplication result
        """
        return self.add.node if self.has_bias else self.mm.node

    @property
    def reshape_in(self) -> Reshape | None:
        """Get the reshape operation before the linear layer, if any.

        Returns:
            Reshape | None: The input reshape node if present, None otherwise
        """
        return Reshape.specialize_from(self.mm.this)

    @property
    def reshape_out(self) -> Reshape | None:
        """Get the reshape operation after the linear layer, if any.

        Returns:
            Reshape | None: The output reshape node if present and unique, None otherwise
        """
        if len(users := list(self.output.users)) != 1:
            return None
        return Reshape.specialize_from(users[0])

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (weight_matmul := WeightMatmul.configure_from(node)):
            return None
        bias_add = None if len(users := list(weight_matmul.mm.users)) != 1 else BiasAdd.configure_from(users[0])
        if bias_add is not None and bias_add.bias.parameter.shape[-1] != weight_matmul.weight.parameter.shape[-1]:
            bias_add = None
        return cls(weight_matmul=weight_matmul, bias_add=bias_add)
