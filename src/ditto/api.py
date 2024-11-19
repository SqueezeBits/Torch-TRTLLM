from collections.abc import Callable

import tensorrt as trt
from loguru import logger
from torch.fx import GraphModule
from torch_tensorrt import dtype
from torch_tensorrt.dynamo._settings import CompilationSettings
from transformers import PreTrainedModel

from ._compile import build_engine, get_inlined_graph_module
from ._export import export
from .arguments_for_export import ArgumentsForExport
from .config import DEFAULT_DEVICE, PassName
from .debug import (
    build_onnx_from_fx,
    open_debug_artifact,
    save_onnx_without_weights,
)
from .pretty_print import detailed_sym_node_str


def trtllm_build(
    model: PreTrainedModel,
    arguments: ArgumentsForExport | None = None,
    *,
    compilation_settings: CompilationSettings | None = None,
    transpose_weights: bool = False,
    mm_in_fp32: bool = False,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    output_names: list[str] | None = None,
    verbose: bool = False,
) -> trt.ICudaEngine:
    try:
        device = next(iter(model.parameters())).device
    except StopIteration:
        logger.warning(f"The model has no parameter. Will set the device as {DEFAULT_DEVICE}")
        device = DEFAULT_DEVICE
    arguments_for_export = arguments or ArgumentsForExport.get_trtllm_inputs(
        device=device,
        use_cache=False,
    )

    if verbose:
        arguments_for_export.print_readable()

    graph_module = trtllm_export(
        model,
        arguments,
        transpose_weights=transpose_weights,
        mm_in_fp32=mm_in_fp32,
        extra_passes=extra_passes,
        verbose=verbose,
    )

    logger.info("Building TensorRT engine ...")
    if isinstance(model, PreTrainedModel) and not output_names:
        logger.info(
            "TRTLLM requires the output name to be 'logits'. Will try to rename the first output name as 'logits'"
        )
        output_names = ["logits"]
    return build_engine(
        graph_module,
        (),
        arguments_for_export.torch_trt_inputs,
        settings=compilation_settings or get_default_compilation_settings(verbose=verbose),
        name=type(model).__name__,
        output_names=output_names,
    )


def get_default_compilation_settings(verbose: bool = False) -> CompilationSettings:
    return CompilationSettings(
        assume_dynamic_shape_support=True,
        enabled_precisions={dtype.f16, dtype.f32},
        debug=verbose,
        optimization_level=3,
        max_aux_streams=-1,
    )


def trtllm_export(
    model: PreTrainedModel,
    arguments: ArgumentsForExport | None = None,
    *,
    transpose_weights: bool = False,
    mm_in_fp32: bool = False,
    skipped_optimizers: list[PassName] | None = None,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    verbose: bool = False,
) -> GraphModule:
    model_name = type(model).__name__
    device = next(iter(model.parameters())).device
    arguments_for_export = arguments or ArgumentsForExport.get_trtllm_inputs(
        device=device,
        use_cache=False,
    )

    if verbose:
        arguments_for_export.print_readable()

    logger.info("torch.exporting module ...")
    exported_program = export(model, arguments_for_export)
    with detailed_sym_node_str(), open(f"{model_name}_program.txt", "w") as f:
        f.write(f"{exported_program}")

    logger.info("Lowering exported program into graph module ...")
    graph_module = get_inlined_graph_module(
        exported_program,
        skipped_optimizers=skipped_optimizers,
        enforce_projections_transposed=transpose_weights,
        enforce_projections_in_fp32=mm_in_fp32,
        extra_passes=extra_passes,
    )

    with detailed_sym_node_str():
        with open_debug_artifact("graph_module.py") as f:
            if f:
                f.write(
                    "\n".join(
                        [
                            "import torch\n",
                            graph_module.print_readable(print_output=False),
                        ]
                    )
                )
        with open_debug_artifact("graph.txt") as f:
            if f:
                f.write(f"{graph_module.graph}")
        with open_debug_artifact("graph_module.onnx", "wb") as f:
            if f:
                save_onnx_without_weights(build_onnx_from_fx(graph_module), f)

    return graph_module
