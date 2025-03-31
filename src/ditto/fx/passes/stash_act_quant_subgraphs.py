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
from typing_extensions import Self

from ...quantization import GlobalQuantConfig, QuantizeMode, QuantizeType, QuantScheme
from ...types import StrictlyTyped
from ..nodes import (
    AddTensorTensor,
    AMax,
    AMin,
    AMinMax,
    ClampScalar,
    ClampTensor,
    DivTensorTensor,
    GetAttr,
    MulTensorTensor,
    SubTensorTensor,
)
from ..subgraphs import Linear
from ..targets import ActivationQuantization
from ..utils import find_nearest
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class StashActQuantSubgraphs(NodewiseOptimizationPass):
    """Match and stash activation quantization subgraphs.

    Attributes:
        global_quant_config (GlobalQuantConfig): The global quantization config.
    """

    global_quant_config: GlobalQuantConfig | None = None

    @property
    def reversed_traversal(self) -> bool:
        return True

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            self.global_quant_config is not None
            and len(self.global_quant_config.quant_configs) == 1
            and (input_quant_scheme := self.global_quant_config.quant_configs[0].input_quant_scheme) is not None
            and input_quant_scheme.bits == 8
            and input_quant_scheme.type == QuantizeType.FLOAT  # Note: only support FP8 quantization for now
            and (activation_quant_config := ActivationQuantConfig.extract_from(node, input_quant_scheme)) is not None
        ):
            return {}

        assert activation_quant_config.scale is None or activation_quant_config.scale.ndim in (
            0,
            1,
        ), "per-token activation quantization is not supported yet"

        activation_quant_config.linear.activation_quantization = ActivationQuantization(
            bits=input_quant_scheme.bits,
            type=input_quant_scheme.type,
            quant_mode=input_quant_scheme.mode,
            scale=activation_quant_config.scale,
            zero_point=activation_quant_config.zero_point,
            dynamic=activation_quant_config.dynamic,
        )

        return {activation_quant_config.mul.node: ReplaceAllUses(by=activation_quant_config.div.this)}


class ActivationQuantConfig(StrictlyTyped):
    """Activation quantization config.

    Attributes:
        linear (Linear): The linear node.
        mul (MulTensorTensor): The mul node.
        div (DivTensorTensor): The div node.
        scale (torch.Tensor | None): The scale tensor.
        zero_point (torch.Tensor | None): The zero point tensor.
        dynamic (bool): Whether the activation quantization is dynamic.
    """

    linear: Linear
    mul: MulTensorTensor
    div: DivTensorTensor
    scale: torch.Tensor | None = None
    zero_point: torch.Tensor | None = None
    dynamic: bool = False

    @classmethod
    def extract_from(cls, node: Node, input_quant_scheme: QuantScheme) -> Self | None:
        """Extract the activation quantization config from the node.

        Args:
            node (Node): The node to extract the activation quantization config from.
            input_quant_scheme (QuantScheme): The input quantization scheme.

        Returns:
            Self | None: The extracted activation quantization config or None
              if no activation quantization config is found.
        """
        if not (
            (linear := Linear.configure_from(node))
            and (
                mul := MulTensorTensor.specialize_from(
                    linear.reshape_in.this if linear.reshape_in else linear.input_node
                )
            )
        ):
            return None

        if input_quant_scheme.mode == QuantizeMode.PER_TENSOR:
            sub = SubTensorTensor.specialize_from(mul.this)
            if (
                sub is None
                and (clamp := ClampTensor.specialize_from(mul.this))
                and (div := DivTensorTensor.specialize_from(clamp.this))
                and (scale := GetAttr.specialize_from(mul.other))
            ):
                return cls(linear=linear, mul=mul, div=div, scale=scale.tensor, zero_point=None)

            if (
                # pylint: disable-next=too-many-boolean-expressions
                sub is not None
                and (clamp := ClampTensor.specialize_from(sub.this))
                and (add := AddTensorTensor.specialize_from(clamp.this))
                and (div := DivTensorTensor.specialize_from(add.this))
                and (scale := GetAttr.specialize_from(mul.other))
                and (zero_point := GetAttr.specialize_from(sub.other))
            ):
                return cls(linear=linear, mul=mul, div=div, scale=scale.tensor, zero_point=zero_point.tensor)

        if (
            # pylint: disable-next=too-many-boolean-expressions
            input_quant_scheme.mode == QuantizeMode.PER_TOKEN
            and (sub := SubTensorTensor.specialize_from(mul.this))
            and (clamp := ClampTensor.specialize_from(sub.this))
            and (add := AddTensorTensor.specialize_from(clamp.this))
            and (div := DivTensorTensor.specialize_from(add.this))
            and (clamp2 := ClampScalar.specialize_from(mul.other))
            and (
                (find_nearest(AMinMax, clamp2.this, follow_first_only=False, max_depth=10) is not None)
                or (
                    find_nearest(AMin, clamp2.this, follow_first_only=False, max_depth=10) is not None
                    and find_nearest(AMax, clamp2.this, follow_first_only=False, max_depth=10) is not None
                )
            )
        ):
            return cls(linear=linear, mul=mul, div=div, dynamic=True)

        return None
