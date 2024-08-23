from typing import Any

import torch
from torch._ops import OpOverload
from torch.export._trace import _export as torch_export
from torch.export.dynamic_shapes import _Dim as DimType
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import _PyTreeCodeGen

from .cache_handler import CacheHandler
from .wrappers import PostExportWrapper, PreExportWrapper


def export(
    cache_handler: CacheHandler,
    model: torch.nn.Module,
    example_inputs: dict[str, Any],
    dynamic_shapes: dict[str, dict[int, DimType] | None],
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
    maintain_input_constraints_checking: bool = False,
) -> PostExportWrapper:
    graph_module = torch_export(
        PreExportWrapper(model, cache_handler=cache_handler),
        (),
        example_inputs,
        dynamic_shapes={"kwargs": dynamic_shapes},
        strict=strict,
        pre_dispatch=pre_dispatch,
    ).module()
    if not maintain_input_constraints_checking:
        graph_module._forward_pre_hooks.clear()
    assert isinstance(graph_module, GraphModule)
    fuse_attention_mask_inputs(graph_module)
    return PostExportWrapper(graph_module, cache_handler=cache_handler)


def fuse_attention_mask_inputs(graph_module: GraphModule) -> None:
    graph = graph_module.graph
    placeholders = get_placeholders(graph)
    if not (
        (prefilled_attention_mask := placeholders.get("prefilled_attention_mask")) is not None
        and (generation_attention_mask := placeholders.get("generation_attention_mask")) is not None
        and len(prefilled_attention_mask.users) == 1
        and len(generation_attention_mask.users) == 1
        and (user := [*prefilled_attention_mask.users][0]) in generation_attention_mask.users
        and isinstance((target := user.target), OpOverload)
        and (target._namespace, target._opname, target._overloadname) == ("aten", "cat", "default")
    ):
        return

    with graph.inserting_after(user):
        attention_mask = graph.placeholder("attention_mask")
        attention_mask.meta = {"val": user.meta["val"], "tensor_meta": user.meta["tensor_meta"]}
    user.replace_all_uses_with(attention_mask)
    graph.erase_node(user)
    graph.erase_node(prefilled_attention_mask)
    graph.erase_node(generation_attention_mask)

    def _modify_names(names: list[str]) -> None:
        names.remove("prefilled_attention_mask")
        names.remove("generation_attention_mask")
        names.append("attention_mask")

    if isinstance((forward_arg_names := graph_module.meta.get("forward_arg_names", None)), list):
        _modify_names(forward_arg_names)

    if isinstance((codegen := graph._codegen), _PyTreeCodeGen) and isinstance(
        (context := codegen.pytree_info.in_spec.children_specs[1].context), list
    ):
        _modify_names(context)

    graph.lint()
    graph.eliminate_dead_code()
    graph_module.recompile()


def get_placeholders(graph: Graph) -> dict[str, Node]:
    return {node.name: node for node in graph.nodes if node.op == "placeholder"}
