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

from collections.abc import Generator

import torch
from torch.fx import GraphModule, Node

from ...quantization import QuantizeMode
from ..nodes import Cat, Dequantize, GetAttr
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class FuseDequantizes(NodewiseOptimizationPass):
    """Fuse consecutive dequantize nodes with cat node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        def name_generator(graph_module: GraphModule) -> Generator[str, None, None]:
            name = "dequantize_fused_constant"
            if not hasattr(graph_module, name):
                yield name

            idx = 1
            while True:
                if not hasattr(graph_module, f"{name}_{idx}"):
                    yield f"{name}_{idx}"
                idx += 1

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
        ):
            return {}

        name_gen = name_generator(node.graph.owning_module)
        input_nodes: list[Node | None] = []

        fused_qweight_tensor = fuse_weights(
            [dequantize.qweight_tensor for dequantize in dequantizes if dequantize.qweight_tensor is not None],
            quant_mode=dequantizes[0].target.mode,
            scales=[dequantize.scale_tensor for dequantize in dequantizes if dequantize.scale_tensor is not None],
        )
        with node.graph.inserting_before(min(dequantize_nodes)):
            fused_qweight_attr = GetAttr.create(node.graph, next(name_gen), fused_qweight_tensor)
        input_nodes.append(fused_qweight_attr.node)

        fused_scale_tensor = fuse_scales(
            [dequantize.scale_tensor for dequantize in dequantizes if dequantize.scale_tensor is not None],
            quant_mode=dequantizes[0].target.mode,
        )
        with node.graph.inserting_before(min(dequantize_nodes)):
            fused_scale_attr = GetAttr.create(node.graph, next(name_gen), fused_scale_tensor)
        input_nodes.append(fused_scale_attr.node)

        if all(dequantize.zeros is not None for dequantize in dequantizes):
            fused_zeros_tensor = torch.cat(
                [dequantize.zeros_tensor for dequantize in dequantizes if dequantize.zeros_tensor is not None],
                dim=-1,
            )
            with node.graph.inserting_before(min(dequantize_nodes)):
                fused_zeros_attr = GetAttr.create(node.graph, next(name_gen), fused_zeros_tensor)
            input_nodes.append(fused_zeros_attr.node)
        else:
            input_nodes.append(None)

        dequantize = dequantizes[0].target.model_copy(update={"output_shape": fused_qweight_tensor.shape})
        with node.graph.inserting_before(node):
            fused_dequantize_node = node.graph.call_function(dequantize, args=tuple(input_nodes))
        nodes_to_replace = [node] + list(dequantize_nodes)
        propagate_metadata_from(*nodes_to_replace, to=fused_dequantize_node)

        return {node: ReplaceAllUses(by=fused_dequantize_node)}


def are_fusible(dequantizes: list[Dequantize]) -> bool:
    """Check if the dequantize nodes are fusible.

    Args:
        dequantizes (list[Dequantize]): A list of dequantize nodes to check for fusibility

    Returns:
        bool: True if all dequantize nodes are fusible, False otherwise
    """
    if not (
        all(dequantizes[0].target.bits == dequantize.target.bits for dequantize in dequantizes[1:])
        and all(dequantizes[0].target.mode == dequantize.target.mode for dequantize in dequantizes[1:])
        and all(dequantizes[0].target.algorithm == dequantize.target.algorithm for dequantize in dequantizes[1:])
        and (qweight_nodes := [dequantize.qweight for dequantize in dequantizes])
        and (scale_nodes := [dequantize.scale for dequantize in dequantizes])
    ):
        return False

    for nodes in (qweight_nodes, scale_nodes):
        if not (
            (attrs := [attr for node in nodes if (attr := GetAttr.specialize_from(node)) is not None])
            and len(attrs) == len(nodes)
            and all(
                attrs[0].tensor.shape[0] == attr.tensor.shape[0] and attrs[0].tensor.dtype == attr.tensor.dtype
                for attr in attrs[1:]
            )
        ):
            return False

    return True


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

    return torch.cat(scales, dim=1)
