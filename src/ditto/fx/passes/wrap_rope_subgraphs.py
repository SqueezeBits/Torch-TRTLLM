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

from loguru import logger
from torch.fx import Node

from ..nodes import Rope
from ..subgraphs import RoPESubgraph
from ..targets import (
    FAKE_ROPE_TARGETS,
    ROPEConfig,
)
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, get_pretrained_config


class WrapRoPESubgraphs(NodewiseOptimizationPass):
    """Match and replace RoPE subgraphs by wrapped RoPE node (required for ReplaceSDPAByGPTAttentionPlugin).

    Attributes:
        has_warned_missing_pretrained_config (bool): Whether to warn about missing pretrained config
    """

    has_warned_missing_pretrained_config: bool = False

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (rope := RoPESubgraph.configure_from(node)):
            return {}

        graph = node.graph
        rope_target = FAKE_ROPE_TARGETS[rope.position_embedding_type]
        with graph.inserting_before(rope.out.node):
            wrapped_rope = Rope.create_with_target(graph, rope_target, rope.x, rope.cos, rope.sin).node

        pretrained_config = get_pretrained_config(graph)
        if not self.has_warned_missing_pretrained_config and pretrained_config is None:
            logger.warning("No pretrained config found in graph module meta data. Default RoPE config will be used.")
            self.has_warned_missing_pretrained_config = True

        rope_config = ROPEConfig.from_pretrained_config(
            pretrained_config,
            positional_embedding_type=rope.position_embedding_type,
            rotary_embedding_dim=rope.rotary_embedding_dim,
        )
        wrapped_rope.meta = rope.out.node.meta
        wrapped_rope.meta["rope_config"] = rope_config

        return {rope.out.node: ReplaceAllUses(by=wrapped_rope)}
