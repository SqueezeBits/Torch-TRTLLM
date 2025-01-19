from tensorrt_llm.functional import PositionEmbeddingType
from torch.fx import Node
from typing_extensions import Self

from ..nodes import AddTensorTensor, Cat, MulTensorTensor, Neg, Slice
from .path import TrailingReformatPath
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
        assert isinstance(self.mul_cos.this, Node) and isinstance(self.mul_cos.other, Node)
        if self.mul_cos.this == self.x:
            return self.mul_cos.other
        return self.mul_cos.this

    @property
    def sin(self) -> Node:
        """The sine embeddings tensor node."""
        assert isinstance(self.mul_sin.this, Node) and isinstance(self.mul_sin.other, Node)
        if Cat.specialize_from(self.mul_sin.this):
            return self.mul_sin.other
        return self.mul_sin.this

    @property
    def position_embedding_type(self) -> PositionEmbeddingType:
        """Get the type of RoPE implementation used in this subgraph.

        Analyzes the slice operations in the subgraph to determine whether this is a GPT-NeoX or GPT-J
        style RoPE implementation based on their slicing patterns:

        Returns:
            PositionEmbeddingType: Either rope_gpt_neox or rope_gptj based on the detected slicing pattern.

        Raises:
            NotImplementedError: If the slicing pattern doesn't match either GPT-NeoX or GPT-J style.
        """
        assert all(isinstance(x, int) for s in [self.slice_1, self.slice_2] for x in [s.start, s.end, s.step])

        if self.slice_1.start == self.slice_2.end and self.slice_1.step == self.slice_2.step == 1:
            return PositionEmbeddingType.rope_gpt_neox

        if self.slice_1.start == self.slice_2.start + 1 and self.slice_1.step == self.slice_2.step == 2:
            return PositionEmbeddingType.rope_gptj

        raise NotImplementedError(f"RoPE type for {self} is not implemented yet.")

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        """Match a RoPE subgraph pattern starting from the given node.

        Args:
            node: The node to start pattern matching from, expected to be an Add operation.

        Returns:
            A configured RoPESubgraph instance if the pattern matches, None otherwise.
        """
        if not (
            (add := AddTensorTensor.specialize_from(node))
            and (mul_lhs := MulTensorTensor.specialize_from(add.this))
            and (mul_rhs := MulTensorTensor.specialize_from(add.other))
        ):
            return None

        if cat := TrailingReformatPath.traceback(Cat, mul_lhs.this) or TrailingReformatPath.traceback(
            Cat, mul_lhs.other
        ):
            mul_cos, mul_sin = mul_rhs, mul_lhs
        elif cat := TrailingReformatPath.traceback(Cat, mul_rhs.this) or TrailingReformatPath.traceback(
            Cat, mul_rhs.other
        ):
            mul_cos, mul_sin = mul_lhs, mul_rhs
        else:
            return None

        if not (
            len(cat.tensors) == 2
            and (neg := TrailingReformatPath.traceback(Neg, cat.tensors[0]))
            and (slice_1 := Slice.specialize_from(neg.this))
            and (slice_2 := TrailingReformatPath.traceback(Slice, cat.tensors[1]))
            and ((slice_1.this == slice_2.this == mul_cos.this) or (slice_1.this == slice_2.this == mul_cos.other))
        ):
            return None

        if not all(isinstance(x, int) for s in [slice_1, slice_2] for x in [s.start, s.end, s.step]):
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
