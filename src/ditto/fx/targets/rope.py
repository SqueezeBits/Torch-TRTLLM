from collections.abc import Callable

import torch
from tensorrt_llm.functional import PositionEmbeddingType
from torch.fx import Graph


def get_llama2_rope_pattern_graph(
    *,
    axis: int = -1,
    embed_dim: int = 128,
) -> Graph:
    graph = Graph()
    x = graph.placeholder("x")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")
    x_cos = graph.call_function(torch.ops.aten.mul.Tensor, (x, cos))
    # Note: integer literals used in the slice nodes will be considered as wild cards by the subgraph matcher
    x_slice_0 = graph.call_function(torch.ops.aten.slice.Tensor, (x, axis, 0, embed_dim // 2))
    x_slice_1 = graph.call_function(torch.ops.aten.slice.Tensor, (x, axis, embed_dim // 2, (1 << 63) - 1))
    neg_x_slice_1 = graph.call_function(torch.ops.aten.neg.default, (x_slice_1,))
    rotated_x = graph.call_function(torch.ops.aten.cat.default, ((neg_x_slice_1, x_slice_0), axis))
    rotated_x_sin = graph.call_function(torch.ops.aten.mul.Tensor, (rotated_x, sin))
    output = graph.call_function(torch.ops.aten.add.Tensor, (x_cos, rotated_x_sin))
    _ = graph.output((output,))
    return graph


def get_llama2_rope_replacment_graph() -> Graph:
    graph = Graph()
    x = graph.placeholder("x")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")
    output = graph.call_function(llama2_rope, (x, cos, sin))
    _ = graph.output((output,))
    return graph


RopeImplType = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
FAKE_ROPE_TARGETS: dict[RopeImplType, PositionEmbeddingType] = {}


def register_rope_target(t: PositionEmbeddingType) -> Callable[[RopeImplType], RopeImplType]:
    def rope_target_wrapper(f: RopeImplType) -> RopeImplType:
        FAKE_ROPE_TARGETS[f] = t
        return f

    return rope_target_wrapper


@register_rope_target(PositionEmbeddingType.rope_gpt_neox)
def llama2_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
