import os
from collections.abc import Callable, Generator

import torch
from loguru import logger
from torch.fx import GraphModule
from torch.fx.graph import CodeGen
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from transformers import PreTrainedModel
from typing_extensions import Buffer

from .arguments import TensorTypeHint, TorchExportArguments, TRTLLMArgumentHint
from .config_gen import generate_trtllm_engine_config
from .configs import (
    TensorRTConfig,
    TRTLLMBuildConfig,
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
from .transform import parallelize, transform
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
    """Build a TensorRT-LLM engine from a PyTorch model.

    Args:
        model (PreTrainedModel): The PyTorch model to convert
        output_dir (str): Directory to save the engine and config files
        profile_config (TRTLLMOptimizationProfileConfig | None): Configuration for optimization profiles
        mapping (TRTLLMMapping | None): Configuration for tensor parallelism mapping
        lora_config (TRTLLMLoraConfig | None): Configuration for LoRA support
        plugin_config (TRTLLMPluginConfig | None): Configuration for TensorRT plugins
        trt_config (TensorRTConfig | None): TensorRT builder configuration
        run_matmuls_in_fp32 (bool): Whether to run matrix multiplications in FP32
        run_activations_in_model_dtype (bool): Whether to run activations in model dtype
        debug_node_names (list[str] | None): List of node names to output for debugging
        engine_cache (BaseEngineCache | None): Cache for TensorRT engines
    """
    network_name = type(model).__name__
    mapping = mapping or TRTLLMMapping()
    plugin_config = plugin_config or TRTLLMPluginConfig.create_from(model.config.torch_dtype)
    profile_config = profile_config or TRTLLMOptimizationProfileConfig.create_from(model.config, plugin_config)
    argument_hint = TRTLLMArgumentHint.configure(profile_config, tp_size=mapping.tp_size)

    logger.info("Exporting the model into graph module and building TensorRT engine")
    graph_generator = trtllm_export(
        model,
        argument_hint,
        model.config.torch_dtype,
        mapping,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        extra_passes=[add_outputs(debug_node_names)] if debug_node_names else None,
    )

    output_names = ["logits"]  # TRTLLM requires the first output name to be 'logits'
    if debug_node_names:
        output_names.extend(debug_node_names)

    for rank, graph_module in graph_generator:
        engine = convert(
            graph_module,
            argument_hint,
            trt_config or TensorRTConfig(),
            engine_cache=engine_cache,
            network_name=network_name,
            output_names=output_names,
            rank=rank,
        )
        logger.opt(lazy=True).debug("Memory Footprint: {m}", m=get_memory_footprint)
        save(engine, f"rank{rank}.engine", output_dir, "Writing serialized engine")

        if rank == 0:
            logger.info("Generating engine config from the optimized graph module")
            engine_config = generate_trtllm_engine_config(
                graph_module,
                TRTLLMBuildConfig.merge(
                    profile_config,
                    TRTLLMModelConfig(
                        lora_config=lora_config or TRTLLMLoraConfig(),
                        plugin_config=plugin_config,
                    ),
                ),
                mapping,
                architecture=network_name,
            )
            save(engine_config.model_dump_json(indent=2), "config.json", output_dir, message="Writing engine config")


def add_outputs(names: list[str]) -> Callable[[GraphModule], GraphModule]:
    """Create a transform pass that adds additional outputs to the graph module.

    Args:
        names (list[str]): List of node names to add as additional outputs

    Returns:
        Callable[[GraphModule], GraphModule]: A callable that transforms a graph module by adding the specified nodes
            as outputs
    """

    def reset_output(gm: GraphModule) -> GraphModule:
        """Add specified nodes as additional outputs to the graph module.

        Args:
            gm (GraphModule): The graph module to modify

        Returns:
            GraphModule: The modified graph module with additional outputs

        Raises:
            RuntimeError: If output node is not found or specified nodes don't exist
        """
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


def save(
    contents: str | Buffer,
    filename: str,
    output_dir: str,
    message: str = "Saving",
) -> None:
    """Save the contents to the specified file.

    Args:
        contents (str | Buffer): The contents to save
        filename (str): The name of the file to save
        output_dir (str): The directory to save the file
        message (str): The message to log
    """

    def get_output_path(filename: str) -> str:
        output_path = os.path.join(output_dir, filename)
        assert not os.path.exists(output_path) or os.path.isfile(output_path)
        if os.path.exists(output_path):
            logger.warning(f"The file at {output_path} will be overwritten")
        return output_path

    filepath = get_output_path(filename)
    mode = "w" if isinstance(contents, str) else "wb"
    logger.info(f"{message} at {filepath}")
    with open(filepath, mode) as f:
        f.write(contents)


def trtllm_export(
    model: PreTrainedModel,
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    mapping: TRTLLMMapping,
    *,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
    skipped_optimizers: list[PassName] | None = None,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    enable_experimental_decompositions: bool = False,
) -> Generator[tuple[int, GraphModule], None, None]:
    """Export a PyTorch model to a graph module and generate TensorRT-LLM engine config.

    Args:
        model (PreTrainedModel): The PyTorch model to export
        argument_hint (TRTLLMArgumentHint): Configuration for input arguments
        dtype (torch.dtype): Data type for the model
        mapping (TRTLLMMapping): Configuration for tensor parallelism mapping
        build_config (TRTLLMBuildConfig): Configuration for building the engine
        run_matmuls_in_fp32 (bool): Whether to run matrix multiplications in FP32
        run_activations_in_model_dtype (bool): Whether to run activations in model dtype
        skipped_optimizers (list[PassName] | None): List of optimization passes to skip
        extra_passes (list[Callable[[GraphModule], GraphModule]] | None): Additional transformation passes to apply
        enable_experimental_decompositions (bool): Whether to enable experimental decompositions

    Returns:
        Generator[tuple[int, GraphModule], None, None]:
            A generator that yields tuples of rank and the transformed graph module
    """
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

    for rank, parallelized_graph_module in parallelize(graph_module, mapping):
        logger.opt(lazy=True).debug("Memory Footprint: {m}", m=lambda: get_memory_footprint(device))
        save_for_debug(f"graph_module_rank{rank}", parallelized_graph_module)
        yield rank, parallelized_graph_module
