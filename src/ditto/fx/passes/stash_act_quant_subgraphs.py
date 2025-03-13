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

from torch.fx.node import Node

from ...quantization import GlobalQuantConfig, QuantizeMode
from ..nodes import ClampTensor, DivTensorTensor, GetAttr, MulTensorTensor
from ..subgraphs import Linear
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
            and self.global_quant_config.input_quant_scheme is not None
            and (linear := Linear.configure_from(node))
            and (
                mul := MulTensorTensor.specialize_from(
                    linear.reshape_in.this if linear.reshape_in else linear.input_node
                )
            )
            and (clamp := ClampTensor.specialize_from(mul.this))
            and (div := DivTensorTensor.specialize_from(clamp.this))
            and (mul_scale := GetAttr.specialize_from(mul.other))
            and (div_scale := GetAttr.specialize_from(div.other))
            and all(mul_scale.tensor == div_scale.tensor)
        ):
            return {}

        assert mul_scale.tensor.ndim in (0, 1), "Only per-tensor quantization is supported currently"
        if self.global_quant_config.input_quant_scheme.mode == QuantizeMode.UNKNOWN:
            self.global_quant_config.input_quant_scheme.mode = QuantizeMode.PER_TENSOR
        linear.activation_quant_scale = mul_scale.tensor
        return {mul.node: ReplaceAllUses(by=div.this)}
