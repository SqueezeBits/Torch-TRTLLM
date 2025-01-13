import os
from collections.abc import Callable

import torch
from loguru import logger
from torch.fx import GraphModule
from torch.fx.graph import CodeGen
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from transformers import PreTrainedModel

from .arguments import TensorTypeHint, TorchExportArguments, TRTLLMArgumentHint
from .config_gen import generate_trtllm_engine_config
from .configs import (
    TensorRTConfig,
    TRTLLMBuildConfig,
    TRTLLMEngineConfig,
    TRTLLMLoraConfig,
    TRTLLMMapping,
    TRTLLMModelConfig,
    TRTLLMOptimizationProfileConfig,
    TRTLLMPluginConfig,
)
from .constants import INPUT_IDS, PassName
from .convert import convert
from .debug import get_memory_footprint, save_for_debug
from .export import export
from .inline import inline
from .transform import transform
from .types import BuiltInConstant


def trtllm_build(
    model: PreTrainedModel,
    output_dir: str,
    *,
    profile_config: TRTLLMOptimizationProfileConfig | None = None,
    mapping: TRTLLMMapping | None = None,
    lora_config: TRTLLMLoraConfig | None = None,
    plugin_config: TRTLLMPluginConfig | None = None,
    trt_config: TensorRTConfig | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
    debug_node_names: list[str] | None = None,
    engine_cache: BaseEngineCache | None = None,
) -> None:
    network_name = type(model).__name__
    model_dtype = model.config.torch_dtype
    mapping = mapping or TRTLLMMapping()
    plugin_config = plugin_config or TRTLLMPluginConfig.create_from(model_dtype)
    profile_config = profile_config or TRTLLMOptimizationProfileConfig.create_from(model.config, plugin_config)
    argument_hint = TRTLLMArgumentHint.configure(profile_config, tp_size=mapping.tp_size)

    logger.info("Exporting the model into graph module")
    graph_module, engine_config = trtllm_export(
        model,
        argument_hint,
        model_dtype,
        mapping,
        TRTLLMBuildConfig.merge(
            profile_config,
            TRTLLMModelConfig(
                lora_config=lora_config or TRTLLMLoraConfig(),
                plugin_config=plugin_config,
            ),
        ),
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        extra_passes=[add_outputs(debug_node_names)] if debug_node_names else None,
    )

    output_names = ["logits"]  # TRTLLM requires the first output name to be 'logits'
    if debug_node_names:
        output_names.extend(debug_node_names)

    logger.info("Building TensorRT engine")
    engine = convert(
        graph_module,
        argument_hint,
        trt_config or TensorRTConfig(),
        engine_cache=engine_cache,
        network_name=network_name,
        output_names=output_names,
    )
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=get_memory_footprint)

    for (name, filename), contents in {
        ("serialized engine", "rank0.engine"): engine,
        ("engine config", "config.json"): engine_config.model_dump_json(indent=2),
    }.items():
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            logger.warning(f"The file at {output_path} will be overwritten")
        logger.info(f"Writing {name} at {output_path}")
        with open(output_path, "w" if isinstance(contents, str) else "wb") as f:
            f.write(contents)


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
    mapping: TRTLLMMapping,
    build_config: TRTLLMBuildConfig,
    *,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
    skipped_optimizers: list[PassName] | None = None,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    enable_experimental_decompositions: bool = False,
) -> tuple[GraphModule, TRTLLMEngineConfig]:
    logger.debug("torch.exporting module")
    hints: dict[str, TensorTypeHint | BuiltInConstant] = {
        INPUT_IDS: argument_hint.batched_input_ids,
        "use_cache": False,
    }
    arguments = TorchExportArguments.from_hints(device=model.device, **hints)
    logger.opt(lazy=True).debug("{x}", x=lambda: arguments)
    exported_program = export(model, arguments)
    save_for_debug("exported_program", exported_program)

    device = model.device
    logger.debug("Lowering exported program into graph module")
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=lambda: get_memory_footprint(device))
    graph_module = inline(
        exported_program,
        class_name=type(model).__name__,
        enable_experimental_decompositions=enable_experimental_decompositions,
    ).cpu()  # Inlined graph module no longer needs to reside on GPU
    # Delete the exported program to free GPU resources
    del exported_program
    torch.cuda.empty_cache()

    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=lambda: get_memory_footprint(device))
    logger.info("Optimizing the graph module")
    graph_module = transform(
        graph_module,
        argument_hint=argument_hint,
        dtype=dtype,
        skipped_optimizers=skipped_optimizers,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        extra_passes=extra_passes,
    )
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=lambda: get_memory_footprint(device))
    save_for_debug("graph_module", graph_module)

    logger.info("Generating engine config from the graph module")
    config = generate_trtllm_engine_config(
        graph_module,
        build_config,
        mapping,
        architecture=type(model).__name__,
    )

    return graph_module, config
