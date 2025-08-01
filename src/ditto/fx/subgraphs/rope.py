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

from tensorrt_llm.functional import PositionEmbeddingType
from torch.fx import Node
from typing_extensions import Self

from ..nodes import AddTensorTensor, Cat, MulTensorTensor, Neg, Slice
from ..utils import get_first_element, get_tensor_metadata
from .path import TrailingReformatPath
from .subgraph import Subgraph


class RoPESubgraph(Subgraph):
    """A subgraph representing a Rotary Position Embedding (RoPE) operation.

    This subgraph identifies the pattern of operations used to apply rotary position embeddings
    to an input tensor.

    Attributes:
        x (Node): The input tensor node for the RoPE operation
        rotary_embedding_dim (int): The dimension of the rotary embeddings
        cos (Node): The cosine component of the rotary embeddings
        sin (Node): The sine component of the rotary embeddings
        slices (tuple[Slice, Slice]): The slicing operations applied in the RoPE computation
        out (AddTensorTensor | Cat): The output node of the RoPE computation,
            which could be an addition or concatenation operation
        is_mrope (bool): Whether the RoPE is for MROPE
    """

    x: Node
    rotary_embedding_dim: int
    cos: Node
    sin: Node
    slices: tuple[Slice, Slice]
    out: AddTensorTensor | Cat
    is_mrope: bool = False

    @property
    def position_embedding_type(self) -> PositionEmbeddingType:
        """Get the type of RoPE implementation used in this subgraph.

        Analyzes the slice operations in the subgraph to determine whether this is a GPT-NeoX or GPT-J
        style RoPE implementation based on their slicing patterns:

        Returns:
            PositionEmbeddingType: Position embedding type of this.

        Raises:
            NotImplementedError: If the slicing pattern doesn't match either GPT-NeoX or GPT-J style.
        """
        if self.is_mrope:
            return PositionEmbeddingType.mrope

        if Slice.are_consecutive(self.slices):
            return PositionEmbeddingType.rope_gpt_neox

        if self.slices[0].start + 1 == self.slices[1].start and self.slices[0].step == self.slices[1].step == 2:
            return PositionEmbeddingType.rope_gptj

        raise NotImplementedError(f"RoPE type for {self} is not implemented yet.")

    @classmethod  # pylint: disable-next=too-many-locals
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

        is_mrope = False
        if (
            cat := TrailingReformatPath.traceback(Cat, mul_lhs.this)
            or TrailingReformatPath.traceback(Cat, mul_lhs.other)
        ) and [n for n in cat.node.all_input_nodes if Neg.specialize_from(n)]:
            mul_cos, mul_sin = mul_rhs, mul_lhs
            if (
                another_cat := TrailingReformatPath.traceback(Cat, mul_rhs.this)
                or TrailingReformatPath.traceback(Cat, mul_rhs.other)
            ) and another_cat is not cat:
                is_mrope = True
        elif (
            cat := TrailingReformatPath.traceback(Cat, mul_rhs.this)
            or TrailingReformatPath.traceback(Cat, mul_rhs.other)
        ) and [n for n in cat.node.all_input_nodes if Neg.specialize_from(n)]:
            mul_cos, mul_sin = mul_lhs, mul_rhs
            if (
                another_cat := TrailingReformatPath.traceback(Cat, mul_lhs.this)
                or TrailingReformatPath.traceback(Cat, mul_lhs.other)
            ) and another_cat is not cat:
                is_mrope = True
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

        if not (rope_emb_meta := get_tensor_metadata(slice_1.this)):
            return None

        x = slice_1.this
        rotary_embedding_dim = rope_emb_meta.shape[-1]
        cos = mul_cos.other if mul_cos.this == slice_1.this else mul_cos.this
        sin = mul_sin.other if Cat.specialize_from(mul_sin.this) else mul_sin.this
        out = add
        # check for non-default rotary_embedding_dim (rotary_embedding_dim != head_size)
        if (  # pylint: disable-next=too-many-boolean-expressions
            len(add.users) == 1
            and (first_user := get_first_element(add.users))
            and (optional_final_cat := Cat.specialize_from(first_user))
            and len(optional_final_cat.tensors) == 2
            and optional_final_cat.tensors[0] == add.node
            and (optional_slice_rope := Slice.specialize_from(slice_1.this))
            and (optional_slice_pass := Slice.specialize_from(optional_final_cat.tensors[1]))
            and Slice.are_consecutive([optional_slice_rope, optional_slice_pass])
        ):
            x = optional_slice_rope.this
            out = optional_final_cat

        return cls(
            x=x,
            rotary_embedding_dim=rotary_embedding_dim,
            cos=cos,
            sin=sin,
            slices=tuple(sorted([slice_1, slice_2], key=lambda s: s.start)),
            out=out,
            is_mrope=is_mrope,
        )
