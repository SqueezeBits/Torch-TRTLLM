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

import os
from collections.abc import Callable, Generator
from typing import TypeAlias

import torch
from loguru import logger
from peft import LoraConfig, PeftModel
from safetensors.torch import save_file as save_as_safetensors
from torch.fx import GraphModule
from torch.fx.graph import CodeGen
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from transformers import PreTrainedModel

from .arguments import TensorTypeHint, TorchExportArguments, TRTLLMArgumentHint
from .configs import (
    TensorRTConfig,
    TRTLLMBuildConfig,
    TRTLLMEngineConfig,
    TRTLLMMapping,
    TRTLLMModelConfig,
    TRTLLMOptimizationProfileConfig,
    TRTLLMPluginConfig,
)
from .constants import INPUT_IDS
from .convert import convert
from .debug import get_memory_footprint, save_for_debug
from .export import export
from .fx import generate_trtllm_engine_config
from .fx.utils import find_output_node
from .inline import inline
from .literals import DTypeLiteral, PassName
from .transform import parallelize, transform
from .types import BuiltInConstant, verify


# pylint: disable=too-many-locals,too-many-arguments
def trtllm_build(
    model: PreTrainedModel | PeftModel,
    output_dir: str,
    *,
    profile_config: TRTLLMOptimizationProfileConfig | None = None,
    mapping: TRTLLMMapping | None = None,
    plugin_config: TRTLLMPluginConfig | None = None,
    trt_config: TensorRTConfig | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
    debug_node_names: list[str] | None = None,
    engine_cache: BaseEngineCache | None = None,
    max_batch_size: int = 256,
    max_seq_len: int | None = None,
    max_num_tokens: int = 8192,
    opt_num_tokens: int | None = None,
    max_beam_width: int = 1,
    logits_dtype: DTypeLiteral = "float32",
    gather_context_logits: bool = False,
    gather_generation_logits: bool = False,
) -> None:
    """Build a TensorRT-LLM engine from a PyTorch model.

    This function performs the following steps:
    1. Configures the necessary parameters and configurations
    2. Exports the PyTorch model to a graph module
    3. Builds TensorRT-LLM engine components
    4. Saves the engine components to the output directory

    Args:
        model (PreTrainedModel | PeftModel): The PyTorch model to convert
        output_dir (str): Directory to save the engine and config files
        profile_config (TRTLLMOptimizationProfileConfig | None): Configuration for optimization profiles
        mapping (TRTLLMMapping | None): Configuration for tensor parallelism mapping
        plugin_config (TRTLLMPluginConfig | None): Configuration for TensorRT plugins
        trt_config (TensorRTConfig | None): TensorRT builder configuration
        run_matmuls_in_fp32 (bool): Whether to run matrix multiplications in FP32
        run_activations_in_model_dtype (bool): Whether to run activations in model dtype
        debug_node_names (list[str] | None): List of node names to output for debugging
        engine_cache (BaseEngineCache | None): Cache for TensorRT engines
        max_batch_size (int): Maximum batch size for TensorRT engine
        max_seq_len (int | None): Maximum sequence length for TensorRT engine
        max_num_tokens (int): Maximum number of tokens for TensorRT engine
        opt_num_tokens (int | None): Optimized number of tokens for TensorRT engine
        max_beam_width (int): Maximum beam width for TensorRT engine
        logits_dtype (DTypeLiteral): Dtype of the output logits
        gather_context_logits (bool): Whether to gather context token logits for benchmark
        gather_generation_logits (bool): Whether to gather generation token logits for benchmark
    """
    mapping = mapping or TRTLLMMapping()
    plugin_config = plugin_config or TRTLLMPluginConfig.create_from(model.config.torch_dtype, mapping.world_size)
    profile_config = profile_config or TRTLLMOptimizationProfileConfig.create_from(
        model.config,
        plugin_config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        opt_num_tokens=opt_num_tokens,
        max_beam_width=max_beam_width,
    )
    model_config = TRTLLMModelConfig(
        plugin_config=plugin_config,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        logits_dtype=logits_dtype,
    )
    argument_hint = TRTLLMArgumentHint.configure(
        profile_config,
        gather_context_logits=gather_context_logits,
        tp_size=mapping.tp_size,
    )

    graph_module = trtllm_export(
        model,
        argument_hint,
        model_config,
        model.config.torch_dtype,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        extra_passes=[add_outputs(debug_node_names)] if debug_node_names else None,
    )

    output_names = ["logits"]  # TRTLLM requires the first output name to be 'logits'
    if debug_node_names:
        output_names.extend(debug_node_names)
    for filename, component in build_trtllm_engine_components(
        graph_module,
        argument_hint,
        build_config=TRTLLMBuildConfig.merge(
            profile_config or TRTLLMOptimizationProfileConfig(),
            TRTLLMModelConfig(plugin_config=plugin_config),
        ),
        mapping=mapping,
        trt_config=trt_config,
        engine_cache=engine_cache,
        network_name=get_network_name(model),
        output_names=output_names,
    ):
        save_component(output_dir, filename, component)


def trtllm_export(
    model: PreTrainedModel | PeftModel,
    argument_hint: TRTLLMArgumentHint,
    model_config: TRTLLMModelConfig,
    dtype: torch.dtype,
    *,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
    skipped_optimizers: list[PassName] | None = None,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
    enable_experimental_decompositions: bool = False,
) -> GraphModule:
    """Export a PyTorch model to a graph module optimized for TensorRT-LLM.

    This function performs several steps:
    1. Exports the PyTorch model to an exported program using torch.export
    2. Inlines the exported program into a graph module
    3. Optimizes the graph module for TensorRT-LLM compatibility

    Args:
        model (PreTrainedModel | PeftModel): The PyTorch model to export
        argument_hint (TRTLLMArgumentHint): Configuration for input arguments
        model_config (TRTLLMModelConfig): Model configurations
        dtype (torch.dtype): Data type for the model
        run_matmuls_in_fp32 (bool, optional): Whether to run matrix multiplications in FP32. Defaults to False.
        run_activations_in_model_dtype (bool, optional): Whether to run activations in model dtype. Defaults to True.
        skipped_optimizers (list[PassName] | None, optional): List of optimization passes to skip. Defaults to None.
        extra_passes (list[Callable[[GraphModule], GraphModule]] | None, optional): Additional transformation passes to
            apply. Defaults to None.
        enable_experimental_decompositions (bool, optional): Whether to enable experimental decompositions.
            Defaults to False.

    Returns:
        GraphModule: The optimized graph module for TensorRT-LLM
    """
    logger.info("Running torch.export")
    hints: dict[str, TensorTypeHint | BuiltInConstant] = {
        INPUT_IDS: argument_hint.batched_input_ids,
        "use_cache": False,
    }
    device = model.device
    arguments = TorchExportArguments.from_hints(device=device, **hints)
    logger.opt(lazy=True).debug("{x}", x=lambda: arguments)
    exported_program = export(model, arguments)
    save_for_debug("exported_program", exported_program)

    logger.info("Inlining the exported program into a graph module")
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=lambda: get_memory_footprint(device))
    graph_module = inline(
        exported_program,
        class_name=get_network_name(model),
        enable_experimental_decompositions=enable_experimental_decompositions,
    ).cpu()  # Inlined graph module no longer needs to reside on GPU
    # Delete the exported program to free GPU resources
    del exported_program
    torch.cuda.empty_cache()

    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=lambda: get_memory_footprint(device))
    logger.info("Optimizing the graph module")
    return transform(
        graph_module,
        argument_hint=argument_hint,
        model_config=model_config,
        dtype=dtype,
        skipped_optimizers=skipped_optimizers,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        extra_passes=extra_passes,
    )


EngineComponent: TypeAlias = TRTLLMEngineConfig | bytes | dict[str, torch.Tensor] | LoraConfig


def build_trtllm_engine_components(
    graph_module: GraphModule,
    argument_hint: TRTLLMArgumentHint,
    *,
    build_config: TRTLLMBuildConfig,
    mapping: TRTLLMMapping | None = None,
    trt_config: TensorRTConfig | None = None,
    engine_cache: BaseEngineCache | None = None,
    network_name: str | None = None,
    output_names: list[str] | None = None,
) -> Generator[tuple[str, EngineComponent], None, None]:
    """Build TensorRT-LLM engine components from a graph module.

    Args:
        graph_module (GraphModule): The graph module to convert to TensorRT engines
        argument_hint (TRTLLMArgumentHint): Configuration for input arguments
        build_config (TRTLLMBuildConfig): Configuration for building TensorRT-LLM engines
        mapping (TRTLLMMapping | None, optional): Tensor parallel mapping configuration. Defaults to None.
        trt_config (TensorRTConfig | None, optional): TensorRT builder configuration. Defaults to None.
        engine_cache (BaseEngineCache | None, optional): Cache for storing/loading built engines. Defaults to None.
        network_name (str | None, optional): Name of the network. Defaults to None.
        output_names (list[str] | None, optional): Names of output tensors. Defaults to None.

    Yields:
        tuple[str, EngineComponent]: A tuple containing:
            - str: The designated filename for the component
            - TRTLLMEngineConfig | bytes | dict[str, torch.Tensor]: Either the engine configuration, serialized engine
                bytes, or Lora state dicts.
    """
    mapping = mapping or TRTLLMMapping()
    for rank, graph_module_per_rank in parallelize(graph_module, mapping):
        if rank == 0:
            yield "config.json", generate_trtllm_engine_config(
                graph_module_per_rank,
                build_config,
                mapping,
                architecture=network_name,
            )
            if (
                lora_state_dicts := verify(
                    graph_module_per_rank.meta.pop("lora_state_dicts", {}),
                    as_type=dict[int, dict[str, torch.Tensor]],
                )
            ) and (
                peft_configs := verify(
                    graph_module_per_rank.meta.pop("peft_configs", {}),
                    as_type=dict[int, LoraConfig],
                )
            ):
                for lora_task_uid, state_dict in lora_state_dicts.items():
                    yield f"lora/{lora_task_uid}/adapter_model.safetensors", state_dict
                for lora_task_uid, lora_config in peft_configs.items():
                    yield f"lora/{lora_task_uid}/adapter_config.json", lora_config
        logger.opt(lazy=True).debug("Memory Footprint: {m}", m=get_memory_footprint)
        save_for_debug(f"graph_module_rank{rank}", graph_module_per_rank)
        logger.info(
            "Building TensorRT engine{for_rank}",
            for_rank=f" for rank {rank}" if mapping.world_size > 1 else "",
        )
        yield f"rank{rank}.engine", convert(
            graph_module_per_rank,
            argument_hint,
            trt_config or TensorRTConfig(),
            engine_cache=engine_cache,
            network_name=network_name,
            output_names=output_names,
            rank=rank,
        )


def save_component(
    engine_dir: str,
    relative_path: str,
    component: EngineComponent,
) -> None:
    """Save a TensorRT-LLM engine component to a file.

    Args:
        engine_dir (str): The root directory of the engine components
        relative_path (str): Relative path to the component within the engine directory
        component (EngineComponent): The component to save, either an engine config, serialized engine bytes, or Lora
            state dicts.
    """
    if os.path.exists(component_path := os.path.join(engine_dir, relative_path)):
        if not os.path.isfile(component_path):
            logger.error(f"The {component_path} is not a regular file")
        else:
            logger.warning(f"The file at {component_path} will be overwritten")

    os.makedirs(os.path.dirname(component_path), exist_ok=True)

    if isinstance(component, TRTLLMEngineConfig):
        logger.info(f"Writing engine config at {component_path}")
        with open(component_path, "w") as f:
            f.write(component.model_dump_json(indent=2))
    elif isinstance(component, LoraConfig):
        logger.info(f"Writing lora config at {component_path}")
        component.save_pretrained(os.path.dirname(component_path))
    elif (lora_state_dict := verify(component, as_type=dict[str, torch.Tensor])) is not None:
        logger.info(f"Writing lora state dict at {component_path}")
        save_as_safetensors(lora_state_dict, component_path)
    elif isinstance(component, bytes):
        logger.info(f"Writing serialized engine at {component_path}")
        with open(component_path, "wb") as f:
            f.write(component)
    else:
        raise ValueError(f"Unsupported component type: {type(component)}")


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
        node = find_output_node(gm)

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


def get_network_name(model: PreTrainedModel | PeftModel) -> str:
    """Get the network name of the model.

    Args:
        model (PreTrainedModel | PeftModel): The model to get the network name from.

    Returns:
        str: The name of the network.
    """
    if isinstance(model, PeftModel):
        return type(model.model).__name__
    return type(model).__name__
