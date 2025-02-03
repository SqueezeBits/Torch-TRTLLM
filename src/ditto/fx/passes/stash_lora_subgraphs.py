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

from ..subgraphs.lora import MultiLora
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class StashLoraSubgraphs(NodewiseOptimizationPass):
    """Match and replace Lora subgraphs by a single Lora plugin node."""

    @property
    def reversed_traversal(self) -> bool:
        return True

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((multi_lora := MultiLora.configure_from(node)) and multi_lora.all_loras_unseen):
            return {}
        multi_lora.set_free_lora_proto()
        return {multi_lora.output_node: ReplaceAllUses(by=multi_lora.pre_lora_output_node)}
