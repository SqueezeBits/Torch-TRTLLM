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
from .configs import TRTLLMMapping
from .constants import PassName
from .contexts import ignore_symbolic_shapes_warning
from .debug import save_for_debug
from .fx import (
    ForgetSubmodules,
    ParallelizeLinear,
    PropagateTensorParallelism,
    ResetCodeGen,
    fake_tensor_prop_on_node_creation,
    get_level1_transform,
    get_optimization_transform,
    update_argument_hint,
)
from .fx.targets import Plugin


def transform(
    graph_module: GraphModule,
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
) -> GraphModule:
    """Transform a PyTorch GraphModule by applying a series of optimization passes.

    Args:
        graph_module (GraphModule): The input PyTorch GraphModule to transform
        argument_hint (TRTLLMArgumentHint): Hints about the arguments for optimization
        dtype (torch.dtype): The target data type for the transformed model
        skipped_optimizers (list[PassName] | None, optional): List of optimizer passes to skip. Defaults to None.
        run_matmuls_in_fp32 (bool, optional): Whether to run matrix multiplications in FP32. Defaults to False.
        run_activations_in_model_dtype (bool, optional): Whether to run activations in model dtype. Defaults to True.
        extra_passes (list[Callable[[GraphModule], GraphModule]] | None, optional): Additional transformation passes to
            apply. Defaults to None.

    Returns:
        GraphModule: The transformed PyTorch GraphModule after applying optimization passes
    """
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
    update_argument_hint(argument_hint, graph_module)

    save_for_debug("initial_graph_module", graph_module)

    custom_pass_manager = DynamoPassManager.build_from_passlist(
        [
            get_optimization_transform(
                argument_hint,
                dtype,
                skipped_optimizers=skipped_optimizers,
                run_matmuls_in_fp32=run_matmuls_in_fp32,
                run_activations_in_model_dtype=run_activations_in_model_dtype,
            ),
            *(extra_passes or []),
        ]
    )
    logger.debug("Running custom passes")
    with fake_tensor_prop_on_node_creation(graph_module), ignore_symbolic_shapes_warning():
        return custom_pass_manager(graph_module)


def parallelize(
    graph_module: GraphModule,
    mapping: TRTLLMMapping,
) -> Generator[tuple[int, GraphModule], None, None]:
    """Parallelize the graph module.

    Args:
        graph_module (GraphModule): The input graph module to parallelize
        mapping (TRTLLMMapping): The mapping of the parallelized graph module

    Returns:
        Generator[tuple[int, GraphModule], None, None]:
            A generator that yields the parallelized graph module for each rank
    """
    if mapping.world_size > 1:
        logger.info("Parallelizing the graph module")
        save_for_debug("graph_module_before_parallelization", graph_module)
        for rank in range(mapping.world_size):
            logger.debug(f"Running parallelize passes for rank {rank}")
            copied_graph_module = copy_graph_module(graph_module)
            mapping_with_rank = mapping.copy_with_rank(rank)
            parallelize_pass_manager = DynamoPassManager.build_from_passlist(
                [
                    PropagateTensorParallelism(mapping=mapping_with_rank).as_transform(),
                    ParallelizeLinear(mapping=mapping_with_rank).as_transform(),
                ]
            )
            with fake_tensor_prop_on_node_creation(copied_graph_module), ignore_symbolic_shapes_warning():
                yield rank, parallelize_pass_manager(copied_graph_module)
    else:
        yield 0, graph_module


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
            # Note: If the target of a node is an instance of Plugin, it isn't copied.
            # We need to copy the Plugin target manually.
            node.target = copy(node.target)
    return copied_graph
