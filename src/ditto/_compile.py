from __future__ import annotations

import logging
import operator
from collections.abc import Callable

import tensorrt as trt
import torch
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo.conversion import (
    CompilationSettings,
    UnsupportedOperatorException,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
    DynamoPassManager,
)
from torch_tensorrt.logging import TRT_LOGGER

from .interpreter import DynamicTRTInterpreter

logger = logging.getLogger(__name__)

CURRENT_DEVICE = Device._current_device()


def build_engine(
    graph_module: GraphModule,
    arg_inputs: tuple[Input, ...],
    kwarg_inputs: dict[str, Input],
    settings: CompilationSettings | None = None,
    engine_cache: BaseEngineCache | None = None,
    name: str | None = None,
) -> trt.ICudaEngine:
    """Convert an graph module to a TensorRT engine."""
    if settings is None:
        settings = CompilationSettings()

    if settings.debug:
        logger.setLevel(logging.DEBUG)

    DYNAMO_CONVERTERS.set_compilation_settings(settings)
    logger.info("Compilation Settings: %s\n", settings)

    flattened_inputs, _ = get_flat_args(arg_inputs, kwarg_inputs)
    output_dtypes = infer_module_output_dtypes(
        graph_module,
        truncate_double=settings.truncate_double,
    )
    try:
        interpreter = DynamicTRTInterpreter(
            graph_module,
            flattened_inputs,
            logger_level=(trt.Logger.VERBOSE if settings.debug else trt.Logger.WARNING),
            output_dtypes=output_dtypes,
            compilation_settings=settings,
            engine_cache=engine_cache,
            network_name=name,
        )
        result = interpreter.run()
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(result.serialized_engine)

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(f"({i}) {name}: {engine.get_tensor_shape(name)}")
        return engine
    except UnsupportedOperatorException as e:
        logger.error(
            f"Conversion of module {graph_module} not currently fully supported or convertible!",
            exc_info=True,
        )
        raise e
    except Exception as e:
        logger.error(
            f"While interpreting the module got an error: {e}",
            exc_info=True,
        )
        raise e


def get_flat_args(
    args: tuple[Input, ...],
    kwargs: dict[str, Input],
) -> tuple[tuple[Input, ...], pytree.TreeSpec]:
    """Flatten args, kwargs using pytree.

    Args:
        args: List[Any] original args passed to __call__
        kwargs: Dict[str, Any] original kwargs passed to __call

    Returns:
        A tuple of (flat_args, received_spec)
        flat_args is flattend args / kwargs
        received_spec is the pytree spec produced while flattening the
        tuple (args, kwargs)
    """
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path((args, kwargs))
    flat_args = tuple(x[1] for x in flat_args_with_path)
    return flat_args, received_spec


def infer_module_output_dtypes(
    module: GraphModule,
    truncate_double: bool = False,
) -> tuple[dtype, ...]:
    the_output = [n for n in module.graph.nodes if n.op == "output"][0]
    output_dtypes: list[dtype] = []
    for node in the_output.all_input_nodes:
        if not isinstance((tensor_meta := node.meta.get("tensor_meta", None)), TensorMetadata):
            raise RuntimeError(f"Found a graph output without tensor metadata: {node.format_node()}")
        output_dtype = tensor_meta.dtype
        output_dtypes.append(
            dtype.float32 if truncate_double and output_dtype == torch.float64 else dtype._from(output_dtype)
        )
    return tuple(output_dtypes)


def get_inlined_graph_module(
    exported_program: ExportedProgram,
    *,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    extra_pre_inline_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    extra_post_inline_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
) -> GraphModule:
    pre_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [*ATEN_PRE_LOWERING_PASSES.passes, *(extra_pre_inline_passes or [])]
    )
    post_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [
            *ATEN_POST_LOWERING_PASSES.passes,
            *(
                eliminate_empty_tensors_from_cat_or_stack,
                eliminate_nop_cat_or_stack,
                # remove_second_outputs_of_scaled_dot_product_attention,
                remove_assert_scalar,
                replace_operator_sub_by_aten_sub,
            ),
            *(extra_post_inline_passes or []),
        ]
    )
    _ = pre_inline_pass_manager(exported_program.graph_module)

    # Decompose the exported program
    exported_program = exported_program.run_decompositions(get_decompositions(enable_experimental_decompositions))
    graph_module = exported_program.module()
    logger.debug(f"Input graph: {graph_module.graph}")

    # Apply lowering on the graph module
    graph_module = post_inline_pass_manager(graph_module)
    logger.debug(f"Lowered Input graph: {graph_module.graph}")

    graph_module._forward_pre_hooks.clear()
    return graph_module


def remove_second_outputs_of_scaled_dot_product_attention(graph_module: GraphModule) -> GraphModule:
    for node in graph_module.graph.nodes:
        if node.target not in (
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
        ):
            continue
        if not (
            len(node.users) == 1
            and (user := list(node.users)[0]).target is operator.getitem
            and len(user.args) == 2
            and user.args[1] == 0
        ):
            print(f"[WARNING] Found a scaled_dot_product_attention node {node} whose second mask output is used")
            continue
        node.target = torch.nn.functional.scaled_dot_product_attention
        user.replace_all_uses_with(node)
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def remove_assert_scalar(graph_module: GraphModule) -> GraphModule:
    nodes_to_remove: list[Node] = []
    for node in graph_module.graph.nodes:
        if node.target is not torch.ops.aten._assert_scalar.default:
            continue
        nodes_to_remove.append(node)
    for node in nodes_to_remove:
        graph_module.graph.erase_node(node)
    return graph_module


def replace_operator_sub_by_aten_sub(graph_module: GraphModule) -> GraphModule:
    for node in graph_module.graph.nodes:
        if not (node.target is operator.sub and len(node.args) == 2):
            continue
        lhs, rhs = node.args
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            node.target = torch.ops.aten.sub.Tensor
        elif isinstance(lhs, torch.Tensor) and isinstance(rhs, bool | complex | float | int):
            node.target = torch.ops.aten.sub.Scalar
        elif isinstance(lhs, bool | complex | float | int) and isinstance(rhs, torch.Tensor):
            node.target = torch.ops.aten.sub.Scalar
            node.args = node.args[::-1]
        elif isinstance(lhs, int) and isinstance(rhs, int):
            node.target = torch.ops.aten.sub.int
        elif isinstance(lhs, float) and isinstance(rhs, float):
            node.target = torch.ops.aten.sub.float
        elif isinstance(lhs, float) and isinstance(rhs, complex):
            node.target = torch.ops.aten.sub.float_complex
        elif isinstance(lhs, complex) and isinstance(rhs, float):
            node.target = torch.ops.aten.sub.complex_float
        elif isinstance(lhs, int) and isinstance(rhs, float):
            node.target = torch.ops.aten.sub.int_float
        elif isinstance(lhs, float) and isinstance(rhs, int):
            node.target = torch.ops.aten.sub.float_int
        else:
            node.target = torch.ops.aten.sub.default
    return graph_module


def eliminate_empty_tensors_from_cat_or_stack(graph_module: GraphModule) -> GraphModule:
    for node in graph_module.graph.nodes:
        if not (
            node.target in (torch.ops.aten.cat.default, torch.ops.aten.stack.default)
            and isinstance((tensors := node.args[0]), list | tuple)
            and all(isinstance(tensor, Node) for tensor in tensors)
        ):
            continue
        non_empty_tensors = tuple(
            tensor
            for tensor in tensors
            if isinstance(tensor, Node)
            and isinstance((tensor_meta := tensor.meta.get("tensor_meta", None)), TensorMetadata)
            and tensor_meta.shape.numel() > 0
        )
        node.args = (non_empty_tensors,) + node.args[1:]
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def eliminate_nop_cat_or_stack(graph_module: GraphModule) -> GraphModule:
    for node in graph_module.graph.nodes:
        if not (
            node.target in (torch.ops.aten.cat.default, torch.ops.aten.stack.default)
            and isinstance((tensors := node.args[0]), list | tuple)
            and len(tensors) == 1
            and isinstance((the_input := tensors[0]), Node)
        ):
            continue
        node.replace_all_uses_with(the_input)
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module
