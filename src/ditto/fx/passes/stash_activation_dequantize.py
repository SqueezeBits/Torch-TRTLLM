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

from torch.fx import Node

from ...quantization import GlobalQuantConfig, QuantizeType
from ..nodes import Dequantize
from ..subgraphs import Linear
from ..targets import ActivationQuantization
from ..utils import get_nodes_with_depth
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class StashActivationDequantize(NodewiseOptimizationPass):
    """Match and stash activation dequantization operations.

    Attributes:
        global_quant_config (GlobalQuantConfig): The global quantization config.
    """

    global_quant_config: GlobalQuantConfig | None = None

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            self.global_quant_config is not None
            and len(self.global_quant_config.quant_configs) == 1
            and (input_quant_scheme := self.global_quant_config.quant_configs[0].input_quant_scheme) is not None
            and input_quant_scheme.bits == 8
            and input_quant_scheme.type == QuantizeType.FLOAT  # Note: only support FP8 quantization for now
            and (dequantize := Dequantize.specialize_from(node))
            and dequantize.bits == 8
            and (
                linears := [
                    linear
                    for n in get_nodes_with_depth(dequantize.node, follow_parent=False, max_depth=2)
                    if (linear := Linear.configure_from(n)) is not None
                ]
            )
            and (len(linears) > 1 or linears[0].weight_dequantize_node != dequantize)
        ):
            return {}

        assert (scale := dequantize.scale_tensor) is None or scale.ndim in (
            0,
            1,
        ), "per-token activation quantization is not supported yet"

        for linear in linears:
            linear.activation_quantization = ActivationQuantization(
                bits=input_quant_scheme.bits,
                type=input_quant_scheme.type,
                quant_mode=input_quant_scheme.mode,
                scale=scale,
                zero_point=dequantize.zeros_tensor,
                dynamic=dequantize.dynamic,
            )

        return {dequantize.node: ReplaceAllUses(by=dequantize.x)}
