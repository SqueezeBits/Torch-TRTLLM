from collections.abc import Callable

import torch
from pydantic import TypeAdapter
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult

from ...arguments import TRTLLMArgumentHint
from ...constants import INPUT_IDS, INPUT_IDS_UNSQUEEZE_DIM
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .graph_pass import GraphOptimizationPass


class AddTRTLLMInputs(GraphOptimizationPass):
    """Add placeholder nodes corresponding to the TRTLLM model inputs."""

    def __init__(self, *, argument_hint: TRTLLMArgumentHint, depth: int = 0) -> None:
        super().__init__(depth=depth)
        self.input_hints = argument_hint.as_dict()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        assert (
            placeholders := {p.name: p for p in graph.find_nodes(op="placeholder")}
        ), "Graph module with no inputs found!"
        last_placeholder = [*placeholders.values()][-1]
        modified = False
        input_ids_unsqueezed = False
        for name, hint in reversed(self.input_hints.items()):
            if name in placeholders:
                if name != INPUT_IDS:
                    continue
                batched_input_ids = get_tensor_metadata(input_ids := placeholders[name])
                with graph.inserting_after(input_ids):
                    unsqueeze = graph.call_function(
                        torch.ops.aten.unsqueeze.default, (input_ids, INPUT_IDS_UNSQUEEZE_DIM)
                    )
                    if batched_input_ids:
                        populate_tensor_metadata(
                            input_ids, batched_input_ids, shape=(batched_input_ids.shape[1 - INPUT_IDS_UNSQUEEZE_DIM],)
                        )
                        populate_tensor_metadata(unsqueeze, batched_input_ids)
                    input_ids.replace_all_uses_with(unsqueeze, delete_user_cb=is_not_equal_to(unsqueeze))
                    modified = True
                    input_ids_unsqueezed = True
                continue
            with graph.inserting_after(last_placeholder):
                placeholder = graph.placeholder(name)
                populate_tensor_metadata(placeholder, shape=hint.symbolic_shape, dtype=hint.dtype)
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
                    squeeze = graph.call_function(torch.ops.aten.squeeze.dim, (output, INPUT_IDS_UNSQUEEZE_DIM))
                if meta := get_tensor_metadata(output):
                    shape = meta.shape[:INPUT_IDS_UNSQUEEZE_DIM] + meta.shape[INPUT_IDS_UNSQUEEZE_DIM + 1 :]
                    populate_tensor_metadata(squeeze, meta, shape=shape)
                replacements[output] = squeeze
            for old_output, new_output in replacements.items():
                the_output_node.replace_input_with(old_output, new_output)
            modified = True

        return PassResult(graph_module, modified)


def is_not_equal_to(target: Node) -> Callable[[Node], bool]:
    def is_not_equal_to_target(user: Node) -> bool:
        return user != target

    return is_not_equal_to_target
