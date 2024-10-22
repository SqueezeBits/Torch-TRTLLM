from __future__ import annotations

import logging
from collections.abc import Callable

import tensorrt as trt
import torch
import torch.utils._pytree as pytree
from tensorrt_llm._common import _is_building
from torch.export import ExportedProgram
from torch.fx import GraphModule
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

from .fx.passes import (
    eliminate_empty_tensors_from_cat_or_stack,
    eliminate_nop_cat_or_stack,
    instantiate_fake_gpt_attention_plugins,
    populate_fake_gpt_attention_plugin_inputs,
    replace_operator_sub_by_aten_sub,
    replace_sdpa_by_fake_gpt_attention_plugin,
)
from .interpreter import DynamicTRTInterpreter

logger = logging.getLogger(__name__)

CURRENT_DEVICE = Device._current_device()


@_is_building  # type: ignore
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
    logger.info(f"Compilation Settings: {settings}\n")

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
    pretrained_config = exported_program.graph_module.meta.get("pretrained_config", None)
    pre_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [*ATEN_PRE_LOWERING_PASSES.passes, *(extra_pre_inline_passes or [])]
    )
    post_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [
            *ATEN_POST_LOWERING_PASSES.passes,
            *(
                eliminate_empty_tensors_from_cat_or_stack,
                eliminate_nop_cat_or_stack,
                replace_operator_sub_by_aten_sub,
                replace_sdpa_by_fake_gpt_attention_plugin,
                instantiate_fake_gpt_attention_plugins,
                populate_fake_gpt_attention_plugin_inputs,
            ),
            *(extra_post_inline_passes or []),
        ]
    )
    _ = pre_inline_pass_manager(exported_program.graph_module)
    exported_program = exported_program.run_decompositions(get_decompositions(enable_experimental_decompositions))
    graph_module = exported_program.module()
    graph_module.meta["pretrained_config"] = pretrained_config
    graph_module = post_inline_pass_manager(graph_module)
    assert isinstance(graph_module, GraphModule)
    graph_module._forward_pre_hooks.clear()
    return graph_module
