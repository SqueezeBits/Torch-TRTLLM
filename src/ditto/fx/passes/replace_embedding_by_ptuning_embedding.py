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

from ..nodes import (
    AddTensorTensor,
    Embedding,
    GetAttr,
    GtScalar,
    MulTensorTensor,
    SubTensorScalar,
    Unsqueeze,
    Where,
)
from ..utils import get_val
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


# pylint: disable=too-many-locals
class ReplaceEmbeddingByPTuningEmbedding(NodewiseOptimizationPass):
    """Replace embedding node by prompt tuning embedding node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (normal_embedding := Embedding.specialize_from(node))
            and (placeholders := {p.name: p for p in node.graph.find_nodes(op="placeholder")})
            and (input_ids := placeholders.get("input_ids"))
            and (prompt_embedding_table := placeholders.get("prompt_embedding_table"))
            and (tasks := placeholders.get("tasks"))
            and (prompt_vocab_size := placeholders.get("prompt_vocab_size"))
        ):
            return {}
        assert isinstance(embedding_weight := get_val(normal_embedding.weight), torch.Tensor)
        vocab_size = embedding_weight.shape[0]

        with node.graph.inserting_before(node):
            gt = GtScalar.create(node.graph, input_ids, vocab_size - 1)
            vocab_size_minus_one = GetAttr.create(
                node.graph, "where_input", torch.tensor([vocab_size - 1], dtype=torch.int32)
            )
            where = Where.create(node.graph, gt, vocab_size_minus_one, input_ids)

        args, kwargs = normal_embedding.args_kwargs(indices=where.node)
        normal_embedding.node.args = args
        normal_embedding.node.kwargs = kwargs

        with node.graph.inserting_before(normal_embedding.node):
            sub = SubTensorScalar.create(node.graph, input_ids, vocab_size)
            zero = GetAttr.create(node.graph, "where2_other", torch.tensor([0], dtype=torch.int32))
            where2 = Where.create(node.graph, gt.node, sub.node, zero)
            mul = MulTensorTensor.create(node.graph, tasks, prompt_vocab_size)
            add = AddTensorTensor.create(node.graph, where2.node, mul.node)
            prompt_embedding = Embedding.create(node.graph, prompt_embedding_table, add.node)
            unsqueeze = Unsqueeze.create(node.graph, gt.node, -1)

        with node.graph.inserting_after(normal_embedding.node):
            where3 = Where.create(node.graph, unsqueeze.node, prompt_embedding.node, normal_embedding.node)

        return {node: ReplaceAllUses(by=where3.node, replace_user_only_if=lambda n: n is not where3.node)}
