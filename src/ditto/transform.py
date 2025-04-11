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

from collections.abc import Callable, Generator
from copy import copy, deepcopy

import torch
from loguru import logger
from torch.fx import GraphModule
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_POST_LOWERING_PASSES,
    DynamoPassManager,
)

from .arguments import TRTLLMArgumentHint
from .configs import TRTLLMModelConfig
from .contexts import ignore_symbolic_shapes_warning
from .debug import get_memory_footprint, save_for_debug
from .fx import (
    ForgetSubmodules,
    ParallelizePipeline,
    Plugin,
    ResetCodeGen,
    fake_tensor_prop_on_node_creation,
    get_level1_transform,
    get_optimization_transform,
    get_preoptimization_transform,
    update_argument_hint,
)
from .literals import PassName
from .quantization import GlobalQuantConfig


# pylint: disable=duplicate-code
def transform(
    graph_module: GraphModule,
    argument_hint: TRTLLMArgumentHint,
    model_config: TRTLLMModelConfig,
    dtype: torch.dtype,
    *,
    global_quant_config: GlobalQuantConfig | None = None,
    skipped_optimizers: list[PassName] | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
    run_routers_in_model_dtype: bool = False,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
) -> Generator[tuple[int, GraphModule], None, None]:
    """Transform a PyTorch GraphModule by applying a series of optimization passes.

    Args:
        graph_module (GraphModule): The input PyTorch GraphModule to transform
        argument_hint (TRTLLMArgumentHint): Hints about the arguments for optimization
        model_config (TRTLLMModelConfig): Model configurations
        dtype (torch.dtype): The target data type for the transformed model
        global_quant_config (GlobalQuantConfig | None, optional): Global quantization config. Defaults to None.
        skipped_optimizers (list[PassName] | None, optional): List of optimizer passes to skip. Defaults to None.
        run_matmuls_in_fp32 (bool, optional): Whether to run matrix multiplications in FP32. Defaults to False.
        run_activations_in_model_dtype (bool, optional): Whether to run activations in model dtype. Defaults to True.
        run_routers_in_model_dtype (bool, optional): Whether to run linear layers for routers in MoE models in model
            dtype instead of FP32. Defaults to False.
        extra_passes (list[Callable[[GraphModule], GraphModule]] | None, optional): Additional transformation passes to
            apply. Defaults to None.

    Returns:
        Generator[tuple[int, GraphModule], None, None]: A generator of tuples containing the rank and the transformed
            GraphModule after applying optimization passes
    """
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=get_memory_footprint)
    logger.info("Optimizing the graph module")
    post_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [
            ResetCodeGen().as_transform(),
            ForgetSubmodules().as_transform(),
            *(
                f
                for f in ATEN_POST_LOWERING_PASSES.passes
                if f.__name__ not in ("constant_fold", "lower_linear", "view_to_reshape")
            ),
            get_level1_transform(skipped_optimizers),
        ]
    )

    logger.debug("Running post-inlining passes")
    with fake_tensor_prop_on_node_creation(graph_module), ignore_symbolic_shapes_warning():
        graph_module = post_inline_pass_manager(graph_module)
    update_argument_hint(argument_hint, graph_module, dtype)

    save_for_debug("initial_graph_module", graph_module)

    for rank in range(argument_hint.mapping.world_size):
        argument_hint.mapping.rank = rank
        copied_graph_module = copy_graph_module(graph_module)
        pre_custom_pass_manager = DynamoPassManager.build_from_passlist(
            [
                get_preoptimization_transform(argument_hint, global_quant_config, dtype),
            ]
        )
        logger.debug(f"Running pre-custom passes for rank {rank}")
        with fake_tensor_prop_on_node_creation(copied_graph_module), ignore_symbolic_shapes_warning():
            copied_graph_module = pre_custom_pass_manager(copied_graph_module)

        save_for_debug(f"preopt_graph_module_rank{rank}", copied_graph_module)

        custom_pass_manager = DynamoPassManager.build_from_passlist(
            [
                get_optimization_transform(
                    argument_hint,
                    model_config,
                    dtype,
                    global_quant_config=global_quant_config,
                    skipped_optimizers=skipped_optimizers,
                    run_matmuls_in_fp32=run_matmuls_in_fp32,
                    run_activations_in_model_dtype=run_activations_in_model_dtype,
                    run_routers_in_model_dtype=run_routers_in_model_dtype,
                ),
                *(extra_passes or []),
            ]
        )
        logger.debug(f"Running custom passes for rank {rank}")
        with fake_tensor_prop_on_node_creation(copied_graph_module), ignore_symbolic_shapes_warning():
            copied_graph_module = custom_pass_manager(copied_graph_module)

        post_pass_manager = DynamoPassManager.build_from_passlist(
            [ParallelizePipeline(argument_hint=argument_hint).as_transform()]
        )
        logger.debug(f"Running post-custom passes for rank {rank}")
        with fake_tensor_prop_on_node_creation(copied_graph_module), ignore_symbolic_shapes_warning():
            yield rank, post_pass_manager(copied_graph_module)


def copy_graph_module(graph_module: GraphModule) -> GraphModule:
    """Copy a graph module and its nodes.

    Args:
        graph_module (GraphModule): The graph module to copy

    Returns:
        GraphModule: The copied graph module
    """
    copied_graph = GraphModule(graph_module.state_dict(), deepcopy(graph_module.graph))
    copied_graph.meta.update(graph_module.meta)
    for node in copied_graph.graph.nodes:
        if isinstance(node.target, Plugin):
            # Note: If the target of a node is an instance of Plugin, it is just moved, not copied.
            # We need to copy the Plugin target manually.
            node.target = copy(node.target)
    return copied_graph
