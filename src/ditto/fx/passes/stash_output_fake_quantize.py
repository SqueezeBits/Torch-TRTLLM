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

from tensorrt_llm.quantization import QuantAlgo
from torch.fx import Node

from ...quantization import GlobalQuantConfig, QuantizeType
from ..nodes import FakeQuantize
from ..subgraphs import Linear
from ..targets import OutputQuantization
from ..utils import get_nodes_with_depth
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class StashOutputFakeQuantize(NodewiseOptimizationPass):
    """Match and stash output fake-quantize operations.

    This pass is used to get output scaling factors for KV cache.

    Attributes:
        global_quant_config (GlobalQuantConfig): The global quantization config.
    """

    global_quant_config: GlobalQuantConfig | None = None

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            self.global_quant_config is not None
            and len(self.global_quant_config.quant_configs) == 1
            and (kv_cache_quant_algo := self.global_quant_config.trtllm_kv_cache_quant_algo) is not None
            and (fake_quantize := FakeQuantize.specialize_from(node))
            and fake_quantize.bits == 8
            and (
                linears := [
                    linear
                    for n in get_nodes_with_depth(fake_quantize.node, follow_parent=True, max_depth=2)
                    if (linear := Linear.configure_from(n)) is not None
                ]
            )
        ):
            return {}

        assert fake_quantize.scale_tensor is not None, "scale tensor is required"

        for linear in linears:
            linear.output_quantization = OutputQuantization(
                bits=8,
                type=QuantizeType.INT if kv_cache_quant_algo == QuantAlgo.INT8 else QuantizeType.FLOAT,
                scale=fake_quantize.scale_tensor,
            )

        return {fake_quantize.node: ReplaceAllUses(by=fake_quantize.x)}
