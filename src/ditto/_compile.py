from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo.conversion import (
    CompilationSettings,
    TRTInterpreterResult,
    UnsupportedOperatorException,
    interpret_module_to_result,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
)
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
    DynamoPassManager,
)
from torch_tensorrt.dynamo.utils import (
    get_flat_args_with_check,
    set_log_level,
    to_torch_tensorrt_device,
)

logger = logging.getLogger(__name__)

CURRENT_DEVICE = Device._current_device()


def compile(
    exported_program: ExportedProgram,
    *,
    arg_inputs: tuple[Input, ...],
    kwarg_inputs: dict[str, Input],
    enabled_precisions: (set[torch.dtype | dtype] | tuple[torch.dtype | dtype]) = _defaults.ENABLED_PRECISIONS,
    debug: bool = _defaults.DEBUG,
    assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
    workspace_size: int = _defaults.WORKSPACE_SIZE,
    min_block_size: int = _defaults.MIN_BLOCK_SIZE,
    torch_executed_ops: set[str] | None = None,
    pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: int | None = _defaults.MAX_AUX_STREAMS,
    version_compatible: bool = _defaults.VERSION_COMPATIBLE,
    optimization_level: int | None = _defaults.OPTIMIZATION_LEVEL,
    use_python_runtime: bool | None = _defaults.USE_PYTHON_RUNTIME,
    truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
    use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    device: Device = CURRENT_DEVICE,
    require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
    disable_tf32: bool = _defaults.DISABLE_TF32,
    sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
    make_refitable: bool = _defaults.MAKE_REFITABLE,
    engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
    num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
    dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
    dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
    dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
    timing_cache_path: str = _defaults.TIMING_CACHE_PATH,
    extra_pre_inline_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    extra_post_inline_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    **kwargs: Any,
) -> TRTInterpreterResult:
    """Convert an ExportedProgram to a TensorRT engine.

    Converts an ExportedProgram to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        exported_program (torch.export.ExportedProgram): Source module

    Keyword Args:
        arg_inputs (tuple[Input, ...]): positional argument specs for TensorRT engine
        kwarg_inputs (tuple[Input, ...]): keyword argument specs for TensorRT engine
        enabled_precisions (Optional[Set[torch.dtype | _enums.dtype]]): The set of datatypes that TensorRT can use
        debug (bool): Whether to print out verbose debugging information
        assume_dynamic_shape_support (bool): Whether to assume dynamic shape support
        workspace_size (int): Workspace TRT is allowed to use for the module (0 is default)
        min_block_size (int): Minimum number of operators per TRT-Engine Block
        torch_executed_ops (Set[str]): Set of operations to run in Torch, regardless of converter coverage
        pass_through_build_failures (bool): Whether to fail on TRT engine build errors (True) or not (False)
        max_aux_streams (Optional[int]): Maximum number of allowed auxiliary TRT streams for each engine
        version_compatible (bool): Provide version forward-compatibility for engine plan files
        optimization_level (Optional[int]): Builder optimization 0-5, higher levels imply longer build time,
            searching for more optimization options. TRT defaults to 3
        use_python_runtime (Optional[bool]): Whether to strictly use Python runtime or C++ runtime. To auto-select a
            runtime based on C++ dependency presence (preferentially choosing C++ runtime if available), leave the
            argument as None
        truncate_double (bool): Whether to truncate float64 TRT engine inputs or weights to float32
        use_fast_partitioner (bool): Whether to use the fast or global graph partitioning system
        enable_experimental_decompositions (bool): Whether to enable all core aten decompositions
            or only a selected subset of them
        device (Device): GPU to compile the model on
        require_full_compilation (bool): Whether to require the graph is fully compiled in TensorRT.
            Only applicable for `ir="dynamo"`; has no effect for `torch.compile` path
        disable_tf32 (bool): Whether to disable TF32 computation for TRT layers
        sparse_weights (bool): Whether to allow the builder to use sparse weights
        make_refitable (bool): Whether to make weights refittable
        refit (bool): Whether to build a refittable engine
        engine_capability (trt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation
        extra_pre_inline_passes (list[Callable[[GraphModule], GraphModule]] | None): extra graph module passes before
            inlining. Defaults to None.
        extra_post_inline_passes (list[Callable[[GraphModule], GraphModule]] | None): extra graph module passes after
            inlining. Defaults to None.
        **kwargs (Any): wild card for handling deprecated keyword arguments

    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    if debug:
        set_log_level(logger.parent, logging.DEBUG)

    if "truncate_long_and_double" in kwargs.keys():
        if truncate_double is not _defaults.TRUNCATE_DOUBLE:
            raise ValueError(
                'Provided configuration for "truncate_double" and deprecated API "truncate_long_and_double", '
                'please only use "truncate_double"'
            )
        else:
            truncate_double = kwargs["truncate_long_and_double"]
            warnings.warn(
                'Compiler option "truncate_long_and_double" is deprecated in favor of "truncate_double" as int64 is '
                "now natively supported, this option will be removed in the next version",
                DeprecationWarning,
                stacklevel=2,
            )
    if "refit" in kwargs.keys():
        warnings.warn(
            "Refit is deprecated. Please use make_refitable=True if you want to enable refitting of the engine.",
            DeprecationWarning,
            stacklevel=2,
        )

    torch_executed_ops = torch_executed_ops if torch_executed_ops is not None else set()

    flattened_input_list = get_flat_args_with_check(exported_program, arg_inputs, kwarg_inputs)[0]

    device = to_torch_tensorrt_device(device)
    enabled_precisions = {dtype._from(e) for e in enabled_precisions}

    compilation_options = {
        "assume_dynamic_shape_support": assume_dynamic_shape_support,
        "enabled_precisions": enabled_precisions,
        "debug": debug,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": torch_executed_ops,
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": use_python_runtime,
        "truncate_double": truncate_double,
        "use_fast_partitioner": use_fast_partitioner,
        "enable_experimental_decompositions": enable_experimental_decompositions,
        "device": device,
        "require_full_compilation": require_full_compilation,
        "disable_tf32": disable_tf32,
        "sparse_weights": sparse_weights,
        "make_refitable": make_refitable,
        "engine_capability": engine_capability,
        "num_avg_timing_iters": num_avg_timing_iters,
        "dla_sram_size": dla_sram_size,
        "dla_local_dram_size": dla_local_dram_size,
        "dla_global_dram_size": dla_global_dram_size,
        "timing_cache_path": timing_cache_path,
    }

    pre_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [*ATEN_PRE_LOWERING_PASSES.passes, *(extra_pre_inline_passes or [])]
    )
    post_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [*ATEN_POST_LOWERING_PASSES.passes, *(extra_post_inline_passes or [])]
    )
    _ = pre_inline_pass_manager(exported_program.graph_module)

    # Decompose the exported program
    exported_program = exported_program.run_decompositions(get_decompositions(enable_experimental_decompositions))
    gm = exported_program.module()
    logger.debug("Input graph: " + str(gm.graph))

    # Apply lowering on the graph module
    gm = post_inline_pass_manager(gm)
    logger.debug("Lowered Input graph: " + str(gm.graph))

    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)

    # Assume converters support dynamic shapes and disable validation
    CONVERTERS.set_dynamic_shape_support(settings.assume_dynamic_shape_support)

    try:
        return interpret_module_to_result(
            gm,
            inputs=flattened_input_list,
            arg_inputs=arg_inputs,
            kwarg_inputs=kwarg_inputs,
            settings=settings,
        )
    except UnsupportedOperatorException as e:
        logger.error(
            f"Conversion of module {gm} not currently fully supported or convertible!",
            exc_info=True,
        )
        raise e
    except Exception as e:
        logger.error(
            f"While interpreting the module got an error: {e}",
            exc_info=True,
        )
        raise e
