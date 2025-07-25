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

from typing import TYPE_CHECKING, Any

import numpy as np
import tensorrt as trt
import torch
from pydantic import PrivateAttr
from tensorrt_llm.functional import QuantMode, SideStreamIDType
from tensorrt_llm.layers.moe import MoeGroupwiseQuantParams, activation_str_to_int_map
from torch.fx import Graph, Node
from typing_extensions import Self

from ...types import StrictlyTyped
from ..nodes import Cat, Permute, Stack
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin

if TYPE_CHECKING:
    from ..subgraphs import MoESubgraph


class MixtureOfExpertsPlugin(Plugin):
    """TensorRT plugin implementation for Mixture of Experts layer.

    Attributes:
        remove_input_padding (bool): Whether to remove padding from input
        number_of_experts (int): Number of expert networks
        experts_per_token (int): Number of experts to route each token to
        expert_hidden_size (int): Hidden dimension size for each expert
        expert_inter_size (int): Intermediate dimension size for each expert
        groupwise_quant_algo (int): Groupwise quantization algorithm
        group_size (int): Group size for groupwise quantization
        activation_type (int): Type of activation function to use
        type_id (trt.DataType): Data type for general tensors
        weight_type_id (trt.DataType): Data type for weight tensors
        quant_mode (int): Quantization mode configuration
        use_final_scales (bool): Whether to use finished states
        use_bias (bool): Whether to use bias terms
        tp_size (int): Tensor parallel size
        tp_rank (int): Tensor parallel rank
        ep_size (int): Expert parallel size
        ep_rank (int): Expert parallel rank
        side_stream_id (int): ID for side stream
        use_lora (bool): Whether to use LoRA
        lora_type_id (trt.DataType): Data type for LoRA weights
        max_low_rank (int): Maximum low-rank for LoRA
    """

    remove_input_padding: bool = True
    number_of_experts: int
    experts_per_token: int
    expert_hidden_size: int
    expert_inter_size: int
    groupwise_quant_algo: int = MoeGroupwiseQuantParams().quant_algo
    group_size: int = -1
    activation_type: int
    type_id: trt.DataType
    weight_type_id: trt.DataType
    quant_mode: int = QuantMode(0)
    use_final_scales: bool = False
    use_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    side_stream_id: int = SideStreamIDType.disable
    use_lora: bool = False
    lora_type_id: trt.DataType = trt.DataType.FLOAT
    max_low_rank: int = 0
    _shared_expert_intermediate_size: int = PrivateAttr(default=0)

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the MoE plugin.

        Args:
            **kwargs: Keyword arguments containing input tensors

        Returns:
            Tensor: Output from MoE layer

        Raises:
            NotImplementedError: If not in fake tensor mode
        """
        if is_in_fake_tensor_mode():
            return kwargs["hidden_states"]
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for a plugin field value.

        Args:
            name (str): Name of the field
            value (Any): Value to get dtype for

        Returns:
            type[np.number]: numpy dtype for the value
        """
        if name in ("remove_input_padding", "use_final_scales", "use_bias", "force_determinism", "use_lora"):
            return np.int32
        return super().get_field_dtype(name, value)


class MixtureOfExpertsPluginInputs(StrictlyTyped):
    """Container for MoE plugin input tensors.

    Attributes:
        hidden_states (Node): Input hidden states
        expert_weights_1 (Node): First set of expert weights
        expert_weights_2 (Node): Second set of expert weights
        token_selected_experts (Node): Index of selected experts
        token_final_scales (Node | None): Final scales for rescaling the output of the experts
        expert_bias_1 (Node | None): First set of expert biases
        expert_bias_2 (Node | None): Second set of expert biases
        hidden_states_raw (Node | None): Raw hidden states for side stream
    """

    hidden_states: Node
    expert_weights_1: Node
    expert_weights_2: Node
    token_selected_experts: Node
    token_final_scales: Node | None = None
    expert_bias_1: Node | None = None
    expert_bias_2: Node | None = None
    # Note: inputs for LoRA are not supported yet
    hidden_states_raw: Node | None = None  # For side stream

    @classmethod
    def create_from(cls, moe: "MoESubgraph", graph: Graph) -> Self:
        """Create plugin inputs from MoE subgraph.

        For the MoE plugin,
        1. the weight tensors of all experts should be stacked together to be fed into
        the plugin, and up-projection weights and gate-projection weights should be concatenated together.
        2. The router logits should be cast to float32.


        Args:
            moe (MoESubgraph): MoE subgraph to extract inputs from
            graph (Graph): Graph to create new nodes in

        Returns:
            MixtureOfExpertsPluginInputs: Container with plugin inputs
        """
        up_weights = Stack.create(graph, [moe.extract_weights(expert)[0] for expert in moe.experts])
        gate_weights = Stack.create(graph, [moe.extract_weights(expert)[1] for expert in moe.experts])
        down_weights = Stack.create(graph, [moe.extract_weights(expert)[2] for expert in moe.experts])
        expert_weights_1 = Cat.create(graph, [up_weights.node, gate_weights.node], -1)
        assert expert_weights_1.ndim == 3
        expert_weights_1 = Permute.create(graph, expert_weights_1, [0, 2, 1])  # type: ignore[assignment]
        expert_weights_2 = Permute.create(graph, down_weights, [0, 2, 1])

        return cls(
            hidden_states=moe.hidden_states,
            expert_weights_1=expert_weights_1.node,
            expert_weights_2=expert_weights_2.node,
            token_selected_experts=moe.token_selected_experts,
            token_final_scales=moe.token_scores,
        )


def get_moe_activation_type() -> int:
    """Get the activation type for MoE plugin.

    Returns:
        int: The activation type enum value
    """
    # TODO: Set activation type for each model.
    # For Qwen MoE models, it is hard-coded to swiglu.
    return activation_str_to_int_map["swiglu"]
