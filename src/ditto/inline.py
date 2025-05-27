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

from loguru import logger
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_PRE_LOWERING_PASSES,
    DynamoPassManager,
)

from .contexts import ignore_symbolic_shapes_warning


def inline(
    exported_program: ExportedProgram,
    *,
    class_name: str | None = None,
    enable_experimental_decompositions: bool = False,
) -> GraphModule:
    """Inline the exported program.

    Args:
        exported_program (ExportedProgram): The exported program to inline.
        class_name (str | None): The name of the class to use for the inline graph module. Defaults to None.
        enable_experimental_decompositions (bool): Whether to enable experimental decompositions. Defaults to False.

    Returns:
        GraphModule: The inline graph module.
    """
    pretrained_config = exported_program.graph_module.meta.get("pretrained_config", None)
    pre_inline_pass_manager = DynamoPassManager.build_from_passlist(ATEN_PRE_LOWERING_PASSES.passes)

    graph_module: GraphModule
    with ignore_symbolic_shapes_warning():
        logger.debug("Running pre-inlining passes")
        _ = pre_inline_pass_manager(exported_program.graph_module, CompilationSettings())
        logger.debug("Running aten decomposition passes")
        exported_program = exported_program.run_decompositions(get_decompositions(enable_experimental_decompositions))
        logger.debug("Inlining the exported program")
        graph_module = exported_program.module()

    graph_module.meta["pretrained_config"] = pretrained_config
    graph_module._forward_pre_hooks.clear()
    if class_name:
        graph_module.__class__.__name__ = class_name
    return graph_module
