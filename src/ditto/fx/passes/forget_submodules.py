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

from ..nodes import GetAttr
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class ForgetSubmodules(NodewiseOptimizationPass):
    """Forget all nested submodules and unnest all nested parameters."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (get_attr := GetAttr.specialize_from(node))
            and "." in get_attr.target
            and (graph_module := (graph := node.graph).owning_module)
        ):
            return {}

        def get_qualname() -> str:
            i = 0
            qualname = "constant"
            while hasattr(graph_module, qualname):
                i += 1
                qualname = f"constant_{i}"
            return qualname

        with graph.inserting_before(node):
            unnested_get_attr = GetAttr.create(graph, get_qualname(), get_attr.parameter)
            inject_stack_trace_from(get_attr, to=unnested_get_attr)
        return {get_attr.node: ReplaceAllUses(by=unnested_get_attr.node)}
