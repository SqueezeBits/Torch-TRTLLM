from collections.abc import Callable

import torch
from loguru import logger
from torch.fx import GraphModule
from torch.fx.graph import CodeGen
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from transformers import PreTrainedModel

from ._convert import convert
from ._export import export
from ._inline import inline
from .arguments import TensorTypeHint, TorchExportArguments, TRTLLMArgumentHint
from .configs import (
    TensorRTConfig,
    TRTLLMBuildConfig,
    TRTLLMEngineConfig,
    TRTLLMModelConfig,
    TRTLLMOptimizationProfileConfig,
    generate_trtllm_pretrained_config,
)
from .constants import DEFAULT_DEVICE, INPUT_IDS, PassName
from .debug import save_for_debug
from .types import BuiltInConstant, DeviceLikeType


def trtllm_build(
    model: PreTrainedModel,
    dtype: torch.dtype,
    *,
    profile_config: TRTLLMOptimizationProfileConfig | None = None,
    model_config: TRTLLMModelConfig | None = None,
    trt_config: TensorRTConfig | None = None,
    allow_matmul_in_fp16: bool = False,
    allow_activation_in_fp16: bool = True,
    debug_node_names: list[str] | None = None,
    device: DeviceLikeType | None = None,
    engine_cache: BaseEngineCache | None = None,
) -> tuple[bytes, TRTLLMEngineConfig]:
    device = _resolve_device(model, device)
    network_name = type(model).__name__
    profile_config = profile_config or TRTLLMOptimizationProfileConfig()
    model_config = model_config or TRTLLMModelConfig()
    trt_config = trt_config or TensorRTConfig()
    argument_hint = TRTLLMArgumentHint.configure(profile_config)

    logger.info("Exporting the model into graph module")
    graph_module = trtllm_export(
        model,
        argument_hint,
        dtype,
        device=device,
        allow_matmul_in_fp16=allow_matmul_in_fp16,
        allow_activation_in_fp16=allow_activation_in_fp16,
        extra_passes=[add_outputs(debug_node_names)] if debug_node_names else None,
    )

    logger.info("Generating engine config from the graph module")
    config = generate_trtllm_engine_config(
        graph_module,
        profile_config,
        model_config,
        architecture=network_name,
    )

    output_names = ["logits"]  # TRTLLM requires the first output name to be 'logits'
    if debug_node_names:
        output_names.extend(debug_node_names)

    logger.info("Building TensorRT engine ...")
    engine = convert(
        graph_module,
        argument_hint,
        trt_config,
        engine_cache=engine_cache,
        network_name=network_name,
        output_names=output_names,
    )

    return engine, config


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
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    *,
    device: DeviceLikeType | None = None,
    allow_matmul_in_fp16: bool = False,
    allow_activation_in_fp16: bool = True,
    skipped_optimizers: list[PassName] | None = None,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
) -> GraphModule:
    logger.debug("torch.exporting module ...")
    device = _resolve_device(model, device)
    hints: dict[str, TensorTypeHint | BuiltInConstant] = {
        INPUT_IDS: argument_hint.batched_input_ids,
        "use_cache": False,
    }
    arguments = TorchExportArguments.from_hints(device=device, **hints)
    logger.opt(lazy=True).debug("{x}", x=lambda: arguments)
    exported_program = export(model, arguments)
    save_for_debug("exported_program", exported_program)

    logger.debug("Lowering exported program into graph module ...")
    graph_module = inline(
        exported_program,
        argument_hint=argument_hint,
        dtype=dtype,
        skipped_optimizers=skipped_optimizers,
        allow_matmul_in_fp16=allow_matmul_in_fp16,
        allow_activation_in_fp16=allow_activation_in_fp16,
        extra_passes=extra_passes,
    )
    save_for_debug("graph_module", graph_module)
    return graph_module


def generate_trtllm_engine_config(
    graph_module: GraphModule,
    profile_config: TRTLLMOptimizationProfileConfig,
    model_config: TRTLLMModelConfig,
    *,
    architecture: str | None = None,
) -> TRTLLMEngineConfig:
    return TRTLLMEngineConfig(
        pretrained_config=generate_trtllm_pretrained_config(
            graph_module,
            architecture=architecture,
        ),
        build_config=TRTLLMBuildConfig.merge(profile_config, model_config),
    )


def _resolve_device(model: torch.nn.Module, device: DeviceLikeType | None) -> DeviceLikeType:
    if device is not None:
        return device
    try:
        return next(iter(model.parameters())).device
    except StopIteration:
        logger.warning(f"The model has no parameter. Will set the device as {DEFAULT_DEVICE}")
    return DEFAULT_DEVICE
