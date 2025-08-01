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

import torch
from torch.fx import Node

from ...quantization import QuantizeMode
from ..nodes import Cat, FakeQuantize, GetAttr
from ..utils import attr_name_generator
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class FuseFakeQuantizes(NodewiseOptimizationPass):
    """Fuse consecutive torch.ops.ditto.fake_quantize.default nodes with cat node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (cat := Cat.specialize_from(node))
            and len(fake_quantize_nodes := cat.tensors) > 1
            and (
                fake_quantizes := [
                    fake_quantize
                    for node in fake_quantize_nodes
                    if (fake_quantize := FakeQuantize.specialize_from(node)) is not None
                ]
            )
            and len(fake_quantizes) == len(fake_quantize_nodes)
            and are_fusible(fake_quantizes)
            and (
                weights := [
                    fake_quantize.input_tensor
                    for fake_quantize in fake_quantizes
                    if fake_quantize.input_tensor is not None
                ]
            )
            and (
                scales := [
                    fake_quantize.scale_tensor
                    for fake_quantize in fake_quantizes
                    if fake_quantize.scale_tensor is not None
                ]
            )
            and (graph_module := node.graph.owning_module) is not None
        ):
            return {}

        name_gen = attr_name_generator(graph_module, "fake_quantize_fused_constant")
        quantize_mode = get_quantize_mode(scales[0].shape)

        fused_weight_tensor = fuse_weights(weights, quant_mode=quantize_mode, scales=scales)
        with node.graph.inserting_before(min(fake_quantize_nodes)):
            fused_weight_attr = GetAttr.create(node.graph, next(name_gen), fused_weight_tensor)

        fused_scale_tensor = fuse_scales(scales, quant_mode=quantize_mode)
        with node.graph.inserting_before(min(fake_quantize_nodes)):
            fused_scale_attr = GetAttr.create(node.graph, next(name_gen), fused_scale_tensor)

        fused_zeros_attr: GetAttr | None = None
        if len(
            zeros := [
                fake_quantize.zeros_tensor for fake_quantize in fake_quantizes if fake_quantize.zeros_tensor is not None
            ]
        ) == len(fake_quantizes):
            fused_zeros_tensor = torch.cat(zeros, dim=-1)
            with node.graph.inserting_before(min(fake_quantize_nodes)):
                fused_zeros_attr = GetAttr.create(node.graph, next(name_gen), fused_zeros_tensor)

        with node.graph.inserting_before(node):
            fused_fake_quantize = FakeQuantize.create(
                node.graph,
                fused_weight_attr,
                fake_quantizes[0].bits,
                fake_quantizes[0].dynamic,
                fake_quantizes[0].output_dtype,
                fused_scale_attr,
                fused_zeros_attr,
                fake_quantizes[0].group_size,
            )
        nodes_to_replace = [node] + list(fake_quantize_nodes)
        propagate_metadata_from(*nodes_to_replace, to=fused_fake_quantize.node)

        return {node: ReplaceAllUses(by=fused_fake_quantize.node)}


def are_fusible(fake_quantizes: list[FakeQuantize]) -> bool:
    """Check if the fake-quantize nodes are fusible.

    Args:
        fake_quantizes (list[FakeQuantize]): A list of fake-quantize nodes to check for fusibility

    Returns:
        bool: True if all fake-quantize nodes are fusible, False otherwise
    """
    if not (
        all(fake_quantizes[0].bits == fake_quantize.bits for fake_quantize in fake_quantizes[1:])
        and all(fake_quantizes[0].dynamic == fake_quantize.dynamic for fake_quantize in fake_quantizes[1:])
        and all(fake_quantizes[0].output_dtype == fake_quantize.output_dtype for fake_quantize in fake_quantizes[1:])
        and (
            weights := [
                fake_quantize.input_tensor for fake_quantize in fake_quantizes if fake_quantize.input_tensor is not None
            ]
        )
        and len(weights) == len(fake_quantizes)
        and all(weights[0].shape[0] == weight.shape[0] and weights[0].dtype == weight.dtype for weight in weights[1:])
        and (
            scales := [
                fake_quantize.scale_tensor for fake_quantize in fake_quantizes if fake_quantize.scale_tensor is not None
            ]
        )
        and len(scales) == len(fake_quantizes)
        and all(get_quantize_mode(scales[0].shape) == get_quantize_mode(scale.shape) for scale in scales[1:])
    ):
        return False

    return True


def get_quantize_mode(scale_shape: torch.Size) -> QuantizeMode:
    """Get the quantize mode from the scale shape.

    Args:
        scale_shape (torch.Size): The shape of the scale tensor
    """
    ndim = len(scale_shape)
    if ndim in (0, 1):
        return QuantizeMode.PER_TENSOR
    if ndim == 2 and (scale_shape[0] == 1 or scale_shape[1] == 1):
        return QuantizeMode.PER_CHANNEL
    return QuantizeMode.PER_GROUP


def fuse_weights(
    weights: list[torch.Tensor],
    *,
    quant_mode: QuantizeMode,
    scales: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Fuse the weights of the fake-quantize nodes.

    Args:
        weights (list[torch.Tensor]): The weights of the fake-quantize nodes
        quant_mode (QuantizeMode): The quantization mode of the fake-quantize nodes
        scales (list[torch.Tensor] | None): The scales of the fake-quantize nodes
    """
    assert len(weights) > 1
    if quant_mode == QuantizeMode.PER_TENSOR:
        assert scales and len(scales) == len(weights)
        if weights[0].dtype == torch.float8_e4m3fn:
            weight_scaling_factor = torch.stack(scales).max(dim=0).values
            fused_weight = torch.cat([weights[i].to(scales[i].dtype) * scales[i] for i in range(len(weights))], dim=1)
            return (fused_weight / weight_scaling_factor).to(torch.float8_e4m3fn)

    return torch.cat(weights, dim=1)


def fuse_scales(scales: list[torch.Tensor], *, quant_mode: QuantizeMode) -> torch.Tensor:
    """Fuse the scales of the fake-quantize nodes.

    Args:
        scales (list[torch.Tensor]): The scales of the fake-quantize nodes
        quant_mode (QuantizeMode): The quantization mode of the fake-quantize nodes
    """
    assert len(scales) > 1
    if quant_mode == QuantizeMode.PER_TENSOR:
        return torch.stack(scales).max(dim=0).values
    if quant_mode == QuantizeMode.PER_CHANNEL:
        return torch.cat(scales, dim=0)

    return torch.cat(scales, dim=1)
