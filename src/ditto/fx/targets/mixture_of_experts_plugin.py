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
from loguru import logger
from pydantic import PrivateAttr
from tensorrt_llm.functional import QuantMode, SideStreamIDType
from tensorrt_llm.layers.moe import MoeConfig as TRTLLMMoeConfig
from tensorrt_llm.layers.moe import activation_str_to_int_map
from torch.fx import Graph, Node
from transformers import PretrainedConfig
from typing_extensions import Self

from ...types import StrictlyTyped
from ..nodes import Cat, Permute, Stack, ToCopy
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin

if TYPE_CHECKING:
    from ..subgraphs import MoESubgraph


class MixtureOfExpertsPlugin(Plugin):
    """TensorRT plugin implementation for Mixture of Experts layer.

    Attributes:
        remove_input_padding (bool): Whether to remove padding from input
        number_of_experts (int): Number of expert networks
        top_k (int): Number of experts to route each token to
        expert_hidden_size (int): Hidden dimension size for each expert
        expert_inter_size (int): Intermediate dimension size for each expert
        activation_type (int): Type of activation function to use
        type_id (trt.DataType): Data type for general tensors
        weight_type_id (trt.DataType): Data type for weight tensors
        output_type_id (trt.DataType): Data type for output tensors
        quant_mode (int): Quantization mode configuration
        use_finished (bool): Whether to use finished states
        use_bias (bool): Whether to use bias terms
        tp_size (int): Tensor parallel size
        tp_rank (int): Tensor parallel rank
        ep_size (int): Expert parallel size
        ep_rank (int): Expert parallel rank
        normalization_mode (int): Mode for normalizing expert routing weights
        sparse_mixer_epsilon (float): Small constant for numerical stability
        force_determinism (bool): Whether to force deterministic behavior
        side_stream_id (int): ID for side stream
        use_lora (bool): Whether to use LoRA
    """

    remove_input_padding: bool = True
    number_of_experts: int
    top_k: int
    expert_hidden_size: int
    expert_inter_size: int
    activation_type: int
    type_id: trt.DataType
    weight_type_id: trt.DataType
    output_type_id: trt.DataType
    quant_mode: int = QuantMode(0)
    use_finished: bool = False
    use_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    normalization_mode: int = TRTLLMMoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
    sparse_mixer_epsilon: float = 0.01
    force_determinism: bool = False
    side_stream_id: int = SideStreamIDType.disable
    use_lora: bool = False
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
        if name in ("remove_input_padding", "use_finished", "use_bias", "force_determinism", "use_lora"):
            return np.int32
        return super().get_field_dtype(name, value)


class MixtureOfExpertsPluginInputs(StrictlyTyped):
    """Container for MoE plugin input tensors.

    Attributes:
        hidden_states (Node): Input hidden states
        routing (Node): Expert routing weights
        expert_weights_1 (Node): First set of expert weights
        expert_weights_2 (Node): Second set of expert weights
        expert_bias_1 (Node | None): First set of expert biases
        expert_bias_2 (Node | None): Second set of expert biases
        finished (Node | None): Finished states
        hidden_states_raw (Node | None): Raw hidden states for side stream
    """

    hidden_states: Node
    routing: Node
    expert_weights_1: Node
    expert_weights_2: Node
    expert_bias_1: Node | None = None
    expert_bias_2: Node | None = None
    finished: Node | None = None
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
        up_weights = Stack.create(graph, [up_weight for (up_weight, _, _) in moe.expert_weights])
        gate_weights = Stack.create(graph, [gate_weight for (_, gate_weight, _) in moe.expert_weights])
        down_weights = Stack.create(graph, [down_weight for (_, _, down_weight) in moe.expert_weights])
        expert_weights_1 = Cat.create(graph, [up_weights, gate_weights], -1)
        assert expert_weights_1.ndim == 3
        expert_weights_1 = Permute.create(graph, expert_weights_1, [0, 2, 1])
        expert_weights_2 = Permute.create(graph, down_weights, [0, 2, 1])

        if moe.router_logits_dtype == torch.bfloat16:
            router_logits = ToCopy.create(graph, moe.router_logits, dtype=torch.float32).node
        elif moe.router_logits_dtype == torch.float32:
            router_logits = moe.router_logits
        else:
            logger.warning(
                f"Router logits with dtype {moe.router_logits_dtype} might not be supported by the MoE plugin. "
                "Add casting to float32 if it's not supported."
            )
            router_logits = moe.router_logits

        return cls(
            hidden_states=moe.hidden_states,
            routing=router_logits,
            expert_weights_1=expert_weights_1.node,
            expert_weights_2=expert_weights_2.node,
        )


# mypy: disable-error-code="union-attr"
# pylint: disable=too-many-return-statements
def get_moe_normalization_mode(pretrained_config: PretrainedConfig | None) -> int:
    """Get the normalization mode for MoE plugin.

    Returns:
        int: The normalization mode enum value
    """
    model_type = pretrained_config.model_type if pretrained_config else None
    # Following the rules defined in tensorrt_llm/models/{model_type}.
    # Default is RENORMALIZE.
    if model_type is None:
        return TRTLLMMoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
    match model_type:
        case "qwen2_moe":
            return TRTLLMMoeConfig.ExpertScaleNormalizationMode.NONE
        case "deepseek_v2":
            if pretrained_config.topk_method == "group_limited_greedy":
                if pretrained_config.num_experts_per_tok > 1 and pretrained_config.norm_topk_prob:
                    return TRTLLMMoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED_RENORM
                return TRTLLMMoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED
            if pretrained_config.topk_method == "greedy":
                if pretrained_config.num_experts_per_tok > 1 and pretrained_config.norm_topk_prob:
                    return TRTLLMMoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
                return TRTLLMMoeConfig.ExpertScaleNormalizationMode.NONE
        case _:
            return TRTLLMMoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
    return TRTLLMMoeConfig.ExpertScaleNormalizationMode.RENORMALIZE


def get_moe_activation_type() -> int:
    """Get the activation type for MoE plugin.

    Returns:
        int: The activation type enum value
    """
    # TODO: Set activation type for each model.
    # For Qwen MoE models, it is hard-coded to swiglu.
    return activation_str_to_int_map["swiglu"]
