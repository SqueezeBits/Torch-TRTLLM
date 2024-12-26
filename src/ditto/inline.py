from loguru import logger
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_PRE_LOWERING_PASSES,
    DynamoPassManager,
)

from .contexts import ignore_symbolic_shapes_warning
from .fx.passes import DecomposeSiLU


def inline(
    exported_program: ExportedProgram,
    *,
    class_name: str | None = None,
    enable_experimental_decompositions: bool = False,
) -> GraphModule:
    pretrained_config = exported_program.graph_module.meta.get("pretrained_config", None)
    pre_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [
            DecomposeSiLU().as_transform(),
            *ATEN_PRE_LOWERING_PASSES.passes,
        ]
    )

    graph_module: GraphModule
    with ignore_symbolic_shapes_warning():
        logger.debug("Running pre-inlining passes")
        _ = pre_inline_pass_manager(exported_program.graph_module)
        logger.debug("Running aten decomposition passes")
        exported_program = exported_program.run_decompositions(get_decompositions(enable_experimental_decompositions))
        logger.debug("Inlining the exported program")
        graph_module = exported_program.module()  # type: ignore[assignment]

    graph_module.meta["pretrained_config"] = pretrained_config
    graph_module._forward_pre_hooks.clear()
    if class_name:
        graph_module.__class__.__name__ = class_name
    return graph_module
