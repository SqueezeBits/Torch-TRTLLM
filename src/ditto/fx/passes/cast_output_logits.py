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
from torch.fx import GraphModule

from ...literals import DTypeLiteral
from ...types import DataType
from ..nodes import ToCopy
from ..utils import find_output_node, get_tensor_metadata
from .infra import GraphOptimizationPass, PassResult


class CastOutputLogits(GraphOptimizationPass):
    """Cast output logits if needed."""

    logits_dtype: DTypeLiteral

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph

        output_node = find_output_node(graph)
        if len(output_node.all_input_nodes) != 1:
            # TODO: handle cases with multiple outputs (eg, medusa, debug outputs)
            return PassResult(graph_module=graph_module, modified=False)

        logits = output_node.all_input_nodes[0]
        target_dtype = DataType(self.logits_dtype).to(torch.dtype)

        if not (output_metadata := get_tensor_metadata(logits)) or output_metadata.dtype == target_dtype:
            return PassResult(graph_module=graph_module, modified=False)

        with graph.inserting_after(logits):
            casted_logits = ToCopy.create(graph, logits, dtype=target_dtype)
            output_node.replace_input_with(logits, casted_logits.node)

        return PassResult(graph_module=graph_module, modified=True)
