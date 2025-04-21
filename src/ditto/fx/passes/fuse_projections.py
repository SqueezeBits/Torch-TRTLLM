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

from abc import abstractmethod
from itertools import accumulate

from torch.fx import Node

from ...constants import MATMUL_FUSION_MAX_OUTPUT_SIZE
from ...literals import LoraPluginInputPrefix
from ..metadata_keys import ACTIVATION_QUANTIZATION, LORA_PREFIX
from ..nodes import MM, AddTensor, Cat, Slice
from ..subgraphs import Linear
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class FuseProjections(NodewiseOptimizationPass):
    """Fuse input projections of a subgraph to a single Linear subgraph."""

    @abstractmethod
    def find_projections(self, node: Node) -> list[Linear]:
        """Find linear projections that can be fused from the given node.

        Args:
            node (Node): The node from which fusible projections can be found

        Returns:
            list[Linear]: List of Linear subgraphs that can be fused
        """

    @property
    @abstractmethod
    def fused_lora_prefix(self) -> LoraPluginInputPrefix | None:
        """The LoRA prefix to assign to the fused projection."""

    @property
    def reversed_traversal(self) -> bool:
        """Whether to traverse nodes in reverse order."""
        return True

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (len(linears := remove_duplicates(self.find_projections(node))) > 1 and are_fusible(linears)):
            return {}

        output_sizes = [user.out_features for user in linears]
        if 0 <= MATMUL_FUSION_MAX_OUTPUT_SIZE < sum(output_sizes):
            return {}

        weight_out_features_dim = linears[0].weight_out_features_dim
        graph = node.graph
        with graph.inserting_before(min(linear.mm.node for linear in linears)):
            # The existing weight nodes must be recreated in order to avoid breaking topological orders.
            weight_nodes = [
                graph.create_node(
                    op=n.op,
                    target=n.target,
                    args=n.args,
                    kwargs=n.kwargs,
                    name=n.name,
                )
                for n in (linear.weight_node for linear in linears)
            ]
            fused_param = Cat.create(graph, weight_nodes, weight_out_features_dim)
            fused_node: Node = MM.create(graph, linears[0].input_node, fused_param).node
            nodes_to_replace = [linear.mm.node for linear in linears]
            propagate_metadata_from(*nodes_to_replace, to=fused_node)
            fused_node.meta[LORA_PREFIX] = self.fused_lora_prefix
            if act_scales := [
                linear.activation_quantization.scale
                for linear in linears
                if linear.activation_quantization and linear.activation_quantization.scale is not None
            ]:
                assert len(act_scales) == len(linears) and all(
                    act_scales[0].item() == act_scale.item() for act_scale in act_scales
                ), "All scale values of target linear layers must be the same"
                assert (
                    activation_quantization := linears[0].activation_quantization
                ) is not None and activation_quantization.zero_point is None, (
                    "fusion of per-tensor activation quantization with zero point is not supported"
                )
                fused_node.meta[ACTIVATION_QUANTIZATION] = activation_quantization.model_copy()

            if all(linear.bias_node is not None for linear in linears):
                # The existing bias nodes must be recreated in order to avoid breaking topological orders.
                bias_nodes = [
                    graph.create_node(
                        op=n.op,
                        target=n.target,
                        args=n.args,
                        kwargs=n.kwargs,
                        name=n.name,
                    )
                    for linear in linears
                    if (n := linear.bias_node) is not None
                ]
                fused_bias_params = Cat.create(graph, bias_nodes)
                fused_node = AddTensor.create(graph, fused_node, fused_bias_params).node
                nodes_to_replace = [linear.add.node for linear in linears if linear.add is not None]
                propagate_metadata_from(*nodes_to_replace, to=fused_node)

            slice_indices = [0, *accumulate(output_sizes)]
            slices = [
                Slice.create(graph, fused_node, weight_out_features_dim, slice_indices[i], slice_indices[i + 1])
                for i in range(len(slice_indices) - 1)
            ]

        results: dict[Node, NodewisePassResult] = {}
        for n, s in zip(nodes_to_replace, slices):
            propagate_metadata_from(n, to=s)
            results[n] = ReplaceAllUses(by=s.node)
        return results


def remove_duplicates(linears: list[Linear]) -> list[Linear]:
    """Filter out duplicate linear layers.

    Args:
        linears (list[Linear]): List of linear layers to filter

    Returns:
        list[Linear]: List containing only unique linear layers
    """
    return list({linear.mm.node: linear for linear in linears}.values())


def are_fusible(linears: list[Linear]) -> bool:
    """Check if the weights of linear layers are fusible.

    Args:
        linears (list[Linear]): A list of linear layers to check for fusibility

    Returns:
        bool: True if all linear layers have fusible weights, False otherwise
    """
    first, *others = linears
    if not (
        all(first.has_transposed_weight == other.has_transposed_weight for other in others)
        and all(first.has_transposed_input == other.has_transposed_input for other in others)
    ):
        return False

    dim = first.weight_in_features_dim
    return all(first.weight_tensor.shape[dim] == other.weight_tensor.shape[dim] for other in others)
