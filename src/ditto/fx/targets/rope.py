from collections.abc import Callable

import torch
from tensorrt_llm.functional import PositionEmbeddingType

from transformers.models.cohere.modeling_cohere import rotate_half as rotate_half_gptj
from transformers.models.llama.modeling_llama import rotate_half as rotate_half_gpt_neox

RopeImplType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
FAKE_ROPE_TARGETS: dict[PositionEmbeddingType, RopeImplType] = {}


def register_rope_target(t: PositionEmbeddingType) -> Callable[[RopeImplType], RopeImplType]:
    """Register a RoPE (Rotary Position Embedding) implementation with its position embedding type.

    Args:
        t (PositionEmbeddingType): The position embedding type to associate with the RoPE implementation

    Returns:
        Callable[[RopeImplType], RopeImplType]: A decorator that registers the RoPE implementation
    """

    def rope_target_wrapper(f: RopeImplType) -> RopeImplType:
        FAKE_ROPE_TARGETS[t] = f
        return f

    return rope_target_wrapper


@register_rope_target(PositionEmbeddingType.rope_gpt_neox)
def rope_gpt_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply gpt_neox style rotary position embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor to apply RoPE to
        cos (torch.Tensor): Cosine component of rotary embeddings
        sin (torch.Tensor): Sine component of rotary embeddings

    Returns:
        torch.Tensor: Input tensor with rotary position embeddings applied
    """
    return (x * cos) + (rotate_half_gpt_neox(x) * sin)


@register_rope_target(PositionEmbeddingType.rope_gptj)
def rope_gptj(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply gptj style rotary position embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor to apply RoPE to
        cos (torch.Tensor): Cosine component of rotary embeddings
        sin (torch.Tensor): Sine component of rotary embeddings

    Returns:
        torch.Tensor: Input tensor with rotary position embeddings applied
    """
    return (x * cos) + (rotate_half_gptj(x) * sin)
