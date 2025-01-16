from torch.fx import Node
from typing_extensions import Self

from ..nodes import AddTensorTensor, Cat, MulTensorTensor, Neg, Slice
from .subgraph import Subgraph


class RoPESubgraph(Subgraph):
    """A subgraph representing a Rotary Position Embedding (RoPE) operation.

    This subgraph identifies the pattern of operations used to apply rotary position embeddings
    to an input tensor.

    Attributes:
        add (AddTensorTensor): The final addition operation
        mul_cos (MulTensorTensor): Multiplication with cosine embeddings
        mul_sin (MulTensorTensor): Multiplication with sine embeddings
        cat (Cat): Concatenation of negated and regular slices
        neg (Neg): Negation operation
        slice_1 (Slice): First slice of input
        slice_2 (Slice): Second slice of input
    """

    add: AddTensorTensor
    mul_cos: MulTensorTensor
    mul_sin: MulTensorTensor
    cat: Cat
    neg: Neg
    slice_1: Slice
    slice_2: Slice

    @property
    def x(self) -> Node:
        """The input tensor node."""
        return self.slice_1.this

    @property
    def cos(self) -> Node:
        """The cosine embeddings tensor node."""
        assert isinstance(self.mul_cos.other, Node)
        return self.mul_cos.other

    @property
    def sin(self) -> Node:
        """The sine embeddings tensor node."""
        assert isinstance(self.mul_sin.other, Node)
        return self.mul_sin.other

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        """Match a RoPE subgraph pattern starting from the given node.

        Args:
            node: The node to start pattern matching from, expected to be an Add operation.

        Returns:
            A configured RoPESubgraph instance if the pattern matches, None otherwise.
        """
        # TODO: Consider commutative property of Binary ops
        if not (
            (add := AddTensorTensor.specialize_from(node))
            and isinstance(add.this, Node)
            and isinstance(add.other, Node)
            and (mul_cos := MulTensorTensor.specialize_from(add.this))
            and (mul_sin := MulTensorTensor.specialize_from(add.other))
            and isinstance(mul_sin.this, Node)
            and (cat := Cat.specialize_from(mul_sin.this))
            and len(cat.tensors) == 2
            and (neg := Neg.specialize_from(cat.tensors[0]))
            and (slice_1 := Slice.specialize_from(neg.this))
            and (slice_2 := Slice.specialize_from(cat.tensors[1]))
            and slice_1.this == slice_2.this == mul_cos.this
        ):
            return None
        return cls(
            add=add,
            mul_cos=mul_cos,
            mul_sin=mul_sin,
            cat=cat,
            neg=neg,
            slice_1=slice_1,
            slice_2=slice_2,
        )
