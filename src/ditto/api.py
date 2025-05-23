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
from .literals import DTypeLiteral
from .quantization import GlobalQuantConfig, resolve_qlinear_device_map
from .transform import transform
from .types import BuiltInConstant, verify


# pylint: disable=too-many-locals,too-many-arguments
def trtllm_build(
    model: PreTrainedModel | PeftModel,
    output_dir: str,
    *,
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
    pp_size: int = 1,
    tp_size: int = 1,
    logits_dtype: DTypeLiteral = "float32",
    gather_context_logits: bool = False,
    gather_generation_logits: bool = False,
    run_routers_in_model_dtype: bool = False,
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
        pp_size (int): N-way pipeline parallelism size
        tp_size (int): N-way tensor parallelism size
        logits_dtype (DTypeLiteral): Dtype of the output logits
        gather_context_logits (bool): Whether to gather context token logits for benchmark
        gather_generation_logits (bool): Whether to gather generation token logits for benchmark
        run_routers_in_model_dtype (bool): Whether to run linear layers for routers in MoE models in model dtype
            instead of FP32.
    """
    mapping = TRTLLMMapping(pp_size=pp_size, tp_size=tp_size)
    plugin_config = TRTLLMPluginConfig.create_from(model.config.torch_dtype, mapping.world_size)
    profile_config = TRTLLMOptimizationProfileConfig.create_from(
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
        mapping=mapping,
    )

    resolve_qlinear_device_map(model)
    global_quant_config = GlobalQuantConfig.create_from(model.config)
    graph_module = trtllm_export(model, argument_hint)

    for rank, transformed_graph_module in transform(
        graph_module,
        argument_hint=argument_hint,
        model_config=model_config,
        dtype=model.config.torch_dtype,
        global_quant_config=global_quant_config,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        run_routers_in_model_dtype=run_routers_in_model_dtype,
        extra_passes=[add_outputs(debug_node_names)] if debug_node_names else None,
    ):
        for filename, component in build_trtllm_engine_components(
            rank,
            transformed_graph_module,
            argument_hint,
            build_config=TRTLLMBuildConfig.merge(profile_config, model_config),
            global_quant_config=global_quant_config,
            trt_config=trt_config,
            engine_cache=engine_cache,
            network_name=get_network_name(model),
            debug_node_names=debug_node_names,
        ):
            save_component(output_dir, filename, component)


def trtllm_export(
    model: PreTrainedModel | PeftModel,
    argument_hint: TRTLLMArgumentHint,
    *,
    enable_experimental_decompositions: bool = False,
) -> GraphModule:
    """Export a PyTorch model to an inlined graph module.

    This function performs several steps:
    1. Exports the PyTorch model to an exported program using torch.export
    2. Inlines the exported program into a graph module

    Args:
        model (PreTrainedModel | PeftModel): The PyTorch model to export
        argument_hint (TRTLLMArgumentHint): Configuration for input arguments
        enable_experimental_decompositions (bool, optional): Whether to enable experimental decompositions.
            Defaults to False.

    Returns:
        GraphModule: The inlined graph module
    """
    logger.info("Running torch.export")
    hints: dict[str, TensorTypeHint | BuiltInConstant] = {
        INPUT_IDS: argument_hint.batched_input_ids,
        "use_cache": False,
    }
    arguments = TorchExportArguments.from_hints(device=model.device, **hints)
    logger.opt(lazy=True).debug("{x}", x=lambda: arguments)
    exported_program = export(model, arguments)
    save_for_debug("exported_program", exported_program)

    logger.info("Inlining the exported program into a graph module")
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=lambda: get_memory_footprint(model.device))
    graph_module = inline(
        exported_program,
        class_name=get_network_name(model),
        enable_experimental_decompositions=enable_experimental_decompositions,
    ).cpu()  # Inlined graph module no longer needs to reside on GPU
    # Delete the exported program to free GPU resources
    del exported_program
    torch.cuda.empty_cache()

    return graph_module


EngineComponent: TypeAlias = TRTLLMEngineConfig | bytes | dict[str, torch.Tensor] | LoraConfig


def build_trtllm_engine_components(
    rank: int,
    graph_module: GraphModule,
    argument_hint: TRTLLMArgumentHint,
    *,
    build_config: TRTLLMBuildConfig,
    global_quant_config: GlobalQuantConfig | None = None,
    trt_config: TensorRTConfig | None = None,
    engine_cache: BaseEngineCache | None = None,
    network_name: str | None = None,
    debug_node_names: list[str] | None = None,
) -> Generator[tuple[str, EngineComponent], None, None]:
    """Build TensorRT-LLM engine components from a graph module.

    Args:
        rank (int): The rank of the engine component
        graph_module (GraphModule): The graph module to convert to TensorRT engines
        argument_hint (TRTLLMArgumentHint): Configuration for input arguments
        build_config (TRTLLMBuildConfig): Configuration for building TensorRT-LLM engines
        global_quant_config (GlobalQuantConfig | None, optional): Global quantization configuration. Defaults to None.
        trt_config (TensorRTConfig | None, optional): TensorRT builder configuration. Defaults to None.
        engine_cache (BaseEngineCache | None, optional): Cache for storing/loading built engines. Defaults to None.
        network_name (str | None, optional): Name of the network. Defaults to None.
        debug_node_names (list[str] | None, optional): List of node names to output for debugging. Defaults to None.

    Yields:
        tuple[str, EngineComponent]: A tuple containing:
            - str: The designated filename for the component
            - TRTLLMEngineConfig | bytes | dict[str, torch.Tensor]: Either the engine configuration, serialized engine
                bytes, or Lora state dicts.
    """
    if rank == 0:
        yield (
            "config.json",
            generate_trtllm_engine_config(
                graph_module,
                build_config,
                argument_hint.mapping,
                global_quant_config=global_quant_config,
                architecture=network_name,
            ),
        )
        if (
            lora_state_dicts := verify(
                graph_module.meta.pop("lora_state_dicts", {}),
                as_type=dict[int, dict[str, torch.Tensor]],
            )
        ) and (
            peft_configs := verify(
                graph_module.meta.pop("peft_configs", {}),
                as_type=dict[int, LoraConfig],
            )
        ):
            for lora_task_uid, state_dict in lora_state_dicts.items():
                yield f"lora/{lora_task_uid}/adapter_model.safetensors", state_dict
            for lora_task_uid, lora_config in peft_configs.items():
                yield f"lora/{lora_task_uid}/adapter_config.json", lora_config
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=get_memory_footprint)
    save_for_debug(f"graph_module_rank{rank}", graph_module)
    logger.info(
        "Building TensorRT engine{for_rank}",
        for_rank=f" for rank {rank}" if argument_hint.mapping.world_size > 1 else "",
    )
    yield (
        f"rank{rank}.engine",
        convert(
            graph_module,
            argument_hint,
            trt_config or TensorRTConfig(),
            engine_cache=engine_cache,
            network_name=network_name,
            debug_node_names=debug_node_names,
        ),
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
