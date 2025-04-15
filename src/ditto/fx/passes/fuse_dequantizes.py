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
from ..nodes import Cat, Dequantize, GetAttr
from ..utils import attr_name_generator
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class FuseDequantizes(NodewiseOptimizationPass):
    """Fuse consecutive torch.ops.ditto.dequantize.default nodes with cat node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (cat := Cat.specialize_from(node))
            and len(dequantize_nodes := cat.tensors) > 1
            and (
                dequantizes := [
                    dequantize
                    for node in dequantize_nodes
                    if (dequantize := Dequantize.specialize_from(node)) is not None
                ]
            )
            and len(dequantizes) == len(dequantize_nodes)
            and are_fusible(dequantizes)
            and (
                weights := [
                    dequantize.weight_tensor for dequantize in dequantizes if dequantize.weight_tensor is not None
                ]
            )
            and (
                scales := [dequantize.scale_tensor for dequantize in dequantizes if dequantize.scale_tensor is not None]
            )
            and (graph_module := node.graph.owning_module) is not None
        ):
            return {}

        name_gen = attr_name_generator(graph_module, "dequantize_fused_constant")
        quantize_mode = get_quantize_mode(scales[0].shape)

        fused_weight_tensor = fuse_weights(weights, quant_mode=quantize_mode, scales=scales)
        with node.graph.inserting_before(min(dequantize_nodes)):
            fused_weight_attr = GetAttr.create(node.graph, next(name_gen), fused_weight_tensor)

        fused_scale_tensor = fuse_scales(scales, quant_mode=quantize_mode)
        with node.graph.inserting_before(min(dequantize_nodes)):
            fused_scale_attr = GetAttr.create(node.graph, next(name_gen), fused_scale_tensor)

        fused_zeros_attr: GetAttr | None = None
        if len(
            zeros := [dequantize.zeros_tensor for dequantize in dequantizes if dequantize.zeros_tensor is not None]
        ) == len(dequantizes):
            fused_zeros_tensor = torch.cat(zeros, dim=-1)
            with node.graph.inserting_before(min(dequantize_nodes)):
                fused_zeros_attr = GetAttr.create(node.graph, next(name_gen), fused_zeros_tensor)

        with node.graph.inserting_before(node):
            fused_dequantize = Dequantize.create(
                node.graph,
                fused_weight_attr,
                fused_scale_attr,
                dequantizes[0].bits,
                fused_zeros_attr,
                dequantizes[0].group_size,
            )
        nodes_to_replace = [node] + list(dequantize_nodes)
        propagate_metadata_from(*nodes_to_replace, to=fused_dequantize.node)

        return {node: ReplaceAllUses(by=fused_dequantize.node)}


def are_fusible(dequantizes: list[Dequantize]) -> bool:
    """Check if the dequantize nodes are fusible.

    Args:
        dequantizes (list[Dequantize]): A list of dequantize nodes to check for fusibility

    Returns:
        bool: True if all dequantize nodes are fusible, False otherwise
    """
    if not (
        all(dequantizes[0].bits == dequantize.bits for dequantize in dequantizes[1:])
        and (
            weights := [dequantize.weight_tensor for dequantize in dequantizes if dequantize.weight_tensor is not None]
        )
        and len(weights) == len(dequantizes)
        and all(weights[0].shape[0] == weight.shape[0] and weights[0].dtype == weight.dtype for weight in weights[1:])
        and (scales := [dequantize.scale_tensor for dequantize in dequantizes if dequantize.scale_tensor is not None])
        and len(scales) == len(dequantizes)
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
    """Fuse the weights of the dequantize nodes.

    Args:
        weights (list[torch.Tensor]): The weights of the dequantize nodes
        quant_mode (QuantizeMode): The quantization mode of the dequantize nodes
        scales (list[torch.Tensor] | None): The scales of the dequantize nodes
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
    """Fuse the scales of the dequantize nodes.

    Args:
        scales (list[torch.Tensor]): The scales of the dequantize nodes
        quant_mode (QuantizeMode): The quantization mode of the dequantize nodes
    """
    assert len(scales) > 1
    if quant_mode == QuantizeMode.PER_TENSOR:
        return torch.stack(scales).max(dim=0).values
    if quant_mode == QuantizeMode.PER_CHANNEL:
        return torch.cat(scales, dim=0)

    return torch.cat(scales, dim=1)
