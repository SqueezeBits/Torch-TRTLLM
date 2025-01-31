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

from ..nodes import MM, ToCopy
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class CastMMToFP32(NodewiseOptimizationPass):
    """Prepend input FP32-castings and append output type-casting for a specific-type matmul node."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (mm := MM.specialize_from(node)) and (output_dtype := mm.output_dtype) and output_dtype != torch.float32
        ):
            return {}
        graph = node.graph
        with graph.inserting_before(node):
            lhs_cast = ToCopy.create(graph, mm.this, dtype=torch.float32)
            rhs_cast = ToCopy.create(graph, mm.other, dtype=torch.float32)
            mm_fp32 = MM.create(graph, lhs_cast, rhs_cast)
            propagate_metadata_from(mm, to=mm_fp32)
            output_cast = ToCopy.create(graph, mm_fp32, dtype=mm.output_dtype)
        return {mm.node: ReplaceAllUses(by=output_cast.node)}
