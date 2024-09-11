import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch.export._trace import _export as torch_export
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import _PyTreeCodeGen
from torch.nn.attention import SDPBackend, sdpa_kernel

from .arguments_for_export import ArgumentsForExport
from .cache_handler import CacheHandler
from .wrappers import PostExportWrapper, PreExportWrapper


def export(
    cache_handler: CacheHandler,
    model: torch.nn.Module,
    arguments: ArgumentsForExport,
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.MATH,
) -> PostExportWrapper:
    with sdpa_kernel(sdp_backends):
        exported_program = torch_export(
            PreExportWrapper(model, cache_handler=cache_handler, constant_inputs=arguments.constant_inputs),
            (),
            arguments.tensor_inputs,
            dynamic_shapes={"kwargs": arguments.constraints},
            strict=strict,
            pre_dispatch=pre_dispatch,
        )

    graph_module = exported_program.module()
    assert isinstance(graph_module, GraphModule)
    graph_module = fuse_attention_mask_inputs(graph_module)
    return PostExportWrapper(graph_module, cache_handler=cache_handler)


def fuse_attention_mask_inputs(graph_module: GraphModule) -> GraphModule:
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
        return graph_module

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

    if isinstance((codegen := graph._codegen), _PyTreeCodeGen):
        target_index: int | None = None
        new_child_spec: pytree.TreeSpec | None = None
        for i, child_spec in enumerate(codegen.pytree_info.in_spec.children_specs):
            if (
                child_spec.type is dict
                and isinstance((context := child_spec.context), list)
                and all(isinstance(x, str) for x in context)
            ):
                target_index = i
                new_context = [*context]
                _modify_names(new_context)
                _, new_child_spec = pytree.tree_flatten_with_path(dict(zip(new_context, new_context)))
                break
        if target_index is not None and new_child_spec is not None:
            codegen.pytree_info.in_spec.children_specs[target_index] = new_child_spec
            graph_module._forward_pre_hooks.clear()

    graph.lint()
    graph.eliminate_dead_code()
    graph_module.recompile()
    return graph_module


def get_placeholders(graph: Graph) -> dict[str, Node]:
    return {node.name: node for node in graph.nodes if node.op == "placeholder"}
