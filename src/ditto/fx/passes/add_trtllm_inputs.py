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

from collections.abc import Callable

import torch
from pydantic import TypeAdapter
from torch._subclasses import FakeTensor
from torch.fx import GraphModule, Node

from ...arguments import TRTLLMArgumentHint
from ...constants import INPUT_IDS, INPUT_IDS_UNSQUEEZE_DIM
from ..nodes import Placeholder, SqueezeDim, Unsqueeze
from .infra import GraphOptimizationPass, PassResult


class AddTRTLLMInputs(GraphOptimizationPass):
    """Add placeholder nodes corresponding to the TRTLLM model inputs.

    Attributes:
        argument_hint (TRTLLMArgumentHint): The argument hint.
    """

    argument_hint: TRTLLMArgumentHint

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        assert (
            placeholders := {p.name: Placeholder._specialize_from(p) for p in graph.find_nodes(op="placeholder")}
        ), "Graph module with no inputs found!"
        last_placeholder = [*placeholders.values()][-1]
        modified = False
        input_ids_unsqueezed = False
        for name, hint in reversed(self.argument_hint.as_dict().items()):
            if hint is None:
                continue
            if name in placeholders:
                if name != INPUT_IDS:
                    continue
                with graph.inserting_after((input_ids := placeholders[name]).node):
                    unsqueeze = Unsqueeze.create(graph, input_ids, INPUT_IDS_UNSQUEEZE_DIM)
                    if isinstance(val := input_ids.output, FakeTensor):
                        with val.fake_mode:
                            input_ids.output = torch.empty(hint.symbolic_shape, dtype=hint.dtype)
                    input_ids.node.replace_all_uses_with(unsqueeze.node, delete_user_cb=is_not_equal_to(unsqueeze.node))
                    modified = True
                    input_ids_unsqueezed = True
                continue
            with graph.inserting_after(last_placeholder.node):
                _ = Placeholder.create(graph, name, hint)
                modified = True

        if (
            input_ids_unsqueezed
            and len(_outputs := graph.find_nodes(op="output")) == 1
            and (
                outputs := TypeAdapter(
                    tuple[Node, ...],
                    config={"arbitrary_types_allowed": True},
                ).validate_python((the_output_node := _outputs[0]).args[0])
            )
        ):
            replacements: dict[Node, Node] = {}
            for output in outputs:
                with graph.inserting_before(the_output_node):
                    squeeze = SqueezeDim.create(graph, output, INPUT_IDS_UNSQUEEZE_DIM)
                replacements[output] = squeeze.node
            for old_output, new_output in replacements.items():
                the_output_node.replace_input_with(old_output, new_output)
            modified = True

        return PassResult(graph_module=graph_module, modified=modified)


def is_not_equal_to(target: Node) -> Callable[[Node], bool]:
    """Check if a node is not equal to a target node.

    Args:
        target (Node): The target node.

    Returns:
        Callable[[Node], bool]: A function that checks if a node is not equal to the target node.
    """

    def is_not_equal_to_target(user: Node) -> bool:
        return user != target

    return is_not_equal_to_target
