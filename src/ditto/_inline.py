import torch
from loguru import logger
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_PRE_LOWERING_PASSES,
    DynamoPassManager,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from .fx.nodes import GetAttr
from .pretty_print import ignore_symbolic_shapes_warning


def inline(
    exported_program: ExportedProgram,
    *,
    class_name: str | None = None,
    enable_experimental_decompositions: bool = False,
) -> GraphModule:
    pretrained_config = exported_program.graph_module.meta.get("pretrained_config", None)
    pre_inline_pass_manager = DynamoPassManager.build_from_passlist(ATEN_PRE_LOWERING_PASSES.passes)

    graph_module: GraphModule
    with ignore_symbolic_shapes_warning():
        logger.debug("Running pre-inlining passes")
        _ = pre_inline_pass_manager(exported_program.graph_module)
        logger.debug("Running aten decomposition passes")
        exported_program = exported_program.run_decompositions(get_decompositions(enable_experimental_decompositions))
        logger.debug("Inlining the exported program")
        graph_module = exported_program.module()  # type: ignore[assignment]

    graph_module = forget_submodules(graph_module)
    graph_module.meta["pretrained_config"] = pretrained_config
    graph_module._forward_pre_hooks.clear()
    if class_name:
        graph_module.__class__.__name__ = class_name
    return graph_module


def forget_submodules(graph_module: GraphModule) -> GraphModule:
    modified = False
    graph = graph_module.graph

    def get_qualname() -> str:
        i = 0
        qualname = "constant"
        while hasattr(graph_module, qualname):
            i += 1
            qualname = f"constant_{i}"
        return qualname

    for node in graph.nodes:
        if not ((get_attr := GetAttr.specialize_from(node)) and "." in get_attr.target):
            continue
        target = get_qualname()
        if isinstance(value := get_attr.parameter, torch.nn.Parameter):
            graph_module.register_parameter(target, value)
        else:
            graph_module.register_buffer(target, value)
        with graph.inserting_before(node):
            unnested_get_attr = graph.get_attr(target)
            unnested_get_attr.meta.update(node.meta)
            unnested_get_attr.stack_trace = (
                f'File "?", line 0, in get_attr\n    {node.name} = self.{(nested_target := get_attr.target)}'
            )
            node.replace_all_uses_with(unnested_get_attr)
            graph.erase_node(node)
            logger.trace(f"Replaced {nested_target} by {target}")
        modified = True

    if modified:
        clean_up_graph_after_modifications(graph_module)

    graph_module._modules.clear()
    return graph_module
