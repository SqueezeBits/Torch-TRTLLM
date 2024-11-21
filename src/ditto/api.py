from collections.abc import Callable

import tensorrt as trt
from loguru import logger
from torch.fx import GraphModule
from torch.fx.graph import CodeGen
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
    allow_matmul_in_fp16: bool = False,
    extra_outputs: list[str] | None = None,
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
        allow_matmul_in_fp16=allow_matmul_in_fp16,
        extra_passes=[add_outputs(extra_outputs)] if extra_outputs else None,
        verbose=verbose,
    )

    logger.info("Building TensorRT engine ...")
    output_names = ["logits"]  # TRTLLM requires the first output name to be 'logits'
    if extra_outputs:
        output_names.extend(extra_outputs)

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
        debug=verbose,
        optimization_level=3,
        max_aux_streams=-1,
    )


def add_outputs(names: list[str]) -> Callable[[GraphModule], GraphModule]:
    def reset_output(gm: GraphModule) -> GraphModule:
        nodes = {n.name: n for n in gm.graph.nodes}
        for node in reversed(gm.graph.nodes):
            if node.op == "output":
                break
        else:
            gm.print_readable()
            raise RuntimeError("No output node found in the graph module")

        try:
            outputs = node.args[0] + tuple(nodes[name] for name in names)
        except KeyError as e:
            gm.print_readable()
            raise RuntimeError(f"Failed to find all of the extra output nodes: {', '.join(names)}") from e

        logger.info(f"Adding new outputs to the graph: {', '.join(names)}")
        node.args = (outputs,)
        gm.graph._codegen = CodeGen()
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm

    return reset_output


def trtllm_export(
    model: PreTrainedModel,
    arguments: ArgumentsForExport | None = None,
    *,
    allow_matmul_in_fp16: bool = False,
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
        allow_matmul_in_fp16=allow_matmul_in_fp16,
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
