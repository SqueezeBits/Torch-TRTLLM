# pylint: disable=unused-argument
from collections.abc import Callable

import torch
from tensorrt_llm.functional import PositionEmbeddingType

from .fake_tensor_mode import is_in_fake_tensor_mode

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
def fake_rope_gpt_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fake gpt_neox style rotary position embedding target.

    Args:
        x (torch.Tensor): Input tensor to apply RoPE to
        cos (torch.Tensor): Cosine component of rotary embeddings
        sin (torch.Tensor): Sine component of rotary embeddings

    Returns:
        torch.Tensor: Input tensor with rotary position embeddings applied


    Raises:
        NotImplementedError: If not in fake tensor mode
    """
    if is_in_fake_tensor_mode():
        return x
    raise NotImplementedError("rope_gpt_neox doesn't have implementation")


@register_rope_target(PositionEmbeddingType.rope_gptj)
def fake_rope_gptj(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fake gptj style rotary position embedding target.

    Args:
        x (torch.Tensor): Input tensor to apply RoPE to
        cos (torch.Tensor): Cosine component of rotary embeddings
        sin (torch.Tensor): Sine component of rotary embeddings

    Returns:
        torch.Tensor: Input tensor with rotary position embeddings applied

    Raises:
        NotImplementedError: If not in fake tensor mode
    """
    if is_in_fake_tensor_mode():
        return x
    raise NotImplementedError("rope_gptj doesn't have implementation")
