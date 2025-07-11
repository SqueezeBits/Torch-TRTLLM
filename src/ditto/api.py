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

import json
import os
from collections.abc import Callable, Generator
from typing import TypeAlias

import modelopt.torch.quantization as mtq
import torch
from loguru import logger
from peft import LoraConfig, PeftModel
from safetensors.torch import save_file as save_as_safetensors
from tensorrt_llm.models import SpeculativeDecodingMode
from tensorrt_llm.quantization import QuantAlgo
from torch.fx import GraphModule
from torch.fx.graph import CodeGen
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from transformers import PreTrainedModel

from .arguments import TensorTypeHint, TorchExportArguments, TRTLLMArgumentHint
from .configs import (
    TensorRTBuilderConfig,
    TensorRTConfig,
    TensorRTNetworkCreationFlags,
    TRTLLMBuildConfig,
    TRTLLMEngineConfig,
    TRTLLMMapping,
    TRTLLMModelConfig,
    TRTLLMOptimizationProfileConfig,
    TRTLLMPluginConfig,
    TRTMultiModalBuildConfig,
)
from .constants import INPUT_IDS
from .convert import convert
from .debug import get_memory_footprint, save_for_debug
from .export import export
from .fx import ReplaceEmbeddingByPTuningEmbedding, generate_trtllm_engine_config
from .fx.utils import find_output_node
from .inline import inline
from .literals import DTypeLiteral, SpeculativeDecodingModeLiteral
from .quantization import GlobalQuantConfig, preprocess_qlinear_module, update_kv_cache_scales
from .transform import multimodal_transform, transform
from .types import BuiltInConstant, verify


def build_multimodal_engine(
    model: PreTrainedModel | PeftModel,
    output_dir: str,
    *,
    input_specs: list[TensorTypeHint],
    max_batch_size: int,
    tp_size: int = 1,
    network_name: str | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    model_type: str = "",
    trt_config: TensorRTConfig | None = None,
) -> None:
    """Build a TensorRT engine for a multimodal model.

    Args:
        model (PreTrainedModel | PeftModel): The PyTorch model to convert
        output_dir (str): Directory to save the engine and config files
        input_specs (list[TensorTypeHint]): List of input specs
        max_batch_size (int): Maximum batch size for TensorRT engine
        tp_size (int): Tensor parallel size
        network_name (str | None): Name of the network. Defaults to None.
        input_names (list[str] | None): List of input names. Defaults to None.
        output_names (list[str] | None): List of output names. Defaults to None.
        model_type (str): Type of the model. Defaults to "".
        trt_config (TensorRTConfig | None): TensorRT builder configuration. Defaults to None.
    """
    if tp_size > 1:
        raise NotImplementedError("Tensor parallel is currently not supported for multimodal models")
    if input_names is None:
        input_names = []
        for i in range(len(input_specs)):
            input_names.append(f"input{i}")
    assert len(input_specs) == len(input_names), "The number of input specs and input names must be the same"

    hints: dict[str, TensorTypeHint] = dict(zip(input_names, input_specs))
    graph_module = export_graph_module(model, hints)
    graph_module = multimodal_transform(graph_module)

    trt_config = trt_config or TensorRTConfig(
        network_creation_flags=TensorRTNetworkCreationFlags(strongly_typed=True),
        builder_config=TensorRTBuilderConfig(),
    )
    builder_config = TRTMultiModalBuildConfig.create_from(
        graph_module,
        trt_config,
        model_name=network_name or "multiModal",
        model_type=model_type,
        dtype=model.dtype,
        max_batch_size=max_batch_size,
        tensor_parallel=tp_size,
    )

    for filename, component in build_multimodal_engine_components(
        graph_module,
        hints,
        output_names=output_names,
        builder_config=builder_config,
        trt_config=trt_config,
        engine_cache=None,
        network_name=network_name or "multiModal",
    ):
        save_component(output_dir, filename, component)


# pylint: disable=too-many-locals,too-many-arguments
def build_llm_engine(
    model: PreTrainedModel | PeftModel,
    output_dir: str,
    *,
    network_name: str | None = None,
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
    tokens_per_block: int = 64,
    use_paged_context_fmha: bool = True,
    run_routers_in_model_dtype: bool = False,
    max_prompt_embedding_table_size: int = 0,
    speculative_decoding_mode: SpeculativeDecodingModeLiteral = "none",
    max_draft_len: int = 0,
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
        network_name (str | None): Name of the network. Defaults to None.
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
        tokens_per_block (int): Number of tokens per block for TensorRT engine
        use_paged_context_fmha (bool): Whether to use paged context FMHA
        run_routers_in_model_dtype (bool): Whether to run linear layers for routers in MoE models in model dtype
            instead of FP32.
        max_prompt_embedding_table_size (int): Maximum size of the prompt embedding table.
        speculative_decoding_mode (SpeculativeDecodingModeLiteral): Mode of speculative decoding.
        max_draft_len (int): Maximum lengths of draft tokens for speculative decoding target model.
    """
    mapping = TRTLLMMapping(pp_size=pp_size, tp_size=tp_size)
    plugin_config = TRTLLMPluginConfig.create_from(
        model.config.torch_dtype,
        mapping.world_size,
        tokens_per_block=tokens_per_block,
        use_paged_context_fmha=use_paged_context_fmha,
    )
    profile_config = TRTLLMOptimizationProfileConfig.create_from(
        model.config,
        plugin_config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        opt_num_tokens=opt_num_tokens,
        max_beam_width=max_beam_width,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        speculative_decoding_mode=SpeculativeDecodingMode.__members__[speculative_decoding_mode.upper()],
        max_draft_len=max_draft_len,
    )
    model_config = TRTLLMModelConfig(
        plugin_config=plugin_config,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        logits_dtype=logits_dtype,
        speculative_decoding_mode=SpeculativeDecodingMode.__members__[speculative_decoding_mode.upper()],
        max_draft_len=max_draft_len,
    )
    argument_hint = TRTLLMArgumentHint.configure(
        profile_config,
        gather_context_logits=gather_context_logits,
        mapping=mapping,
    )

    if (global_quant_config := GlobalQuantConfig.create_from(model)) is not None:
        preprocess_qlinear_module(model, global_quant_config.quant_method)
        update_kv_cache_scales(model, global_quant_config.quant_method, global_quant_config.trtllm_kv_cache_quant_algo)

    if (
        global_quant_config is not None
        and global_quant_config.trtllm_kv_cache_quant_algo in (QuantAlgo.INT8, QuantAlgo.FP8)
        and plugin_config.use_paged_context_fmha
    ):
        logger.warning(
            "Paged Context FMHA is not compatible with int8/fp8 KV cache. "
            "Enabling it may lead to incorrect results or even a crash. "
            "To disable Paged Context FMHA, use `--no-use-paged-context-fmha`."
        )

    hints: dict[str, TensorTypeHint | BuiltInConstant] = {
        INPUT_IDS: argument_hint.batched_input_ids,
        "use_cache": False,
    }
    with mtq.utils.export_torch_mode():
        graph_module = export_graph_module(model, hints)

    extra_passes: list[Callable[[GraphModule, CompilationSettings], GraphModule]] = []
    if debug_node_names:
        extra_passes.append(add_outputs(debug_node_names))
    if max_prompt_embedding_table_size > 0:
        extra_passes.append(ReplaceEmbeddingByPTuningEmbedding().as_transform())

    for rank, transformed_graph_module in transform(
        graph_module,
        argument_hint=argument_hint,
        model_config=model_config,
        dtype=model.config.torch_dtype,
        global_quant_config=global_quant_config,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        run_routers_in_model_dtype=run_routers_in_model_dtype,
        extra_passes=extra_passes,
    ):
        for filename, component in build_trtllm_engine_components(
            rank,
            transformed_graph_module,
            argument_hint,
            build_config=TRTLLMBuildConfig.merge(profile_config, model_config),
            global_quant_config=global_quant_config,
            trt_config=trt_config,
            engine_cache=engine_cache,
            network_name=network_name or get_network_name(model),
            debug_node_names=debug_node_names,
        ):
            save_component(output_dir, filename, component)


def export_graph_module(
    model: PreTrainedModel | PeftModel,
    hints: dict[str, TensorTypeHint],
    *,
    enable_experimental_decompositions: bool = False,
) -> GraphModule:
    """Export a PyTorch model to an inlined graph module.

    This function performs several steps:
    1. Exports the PyTorch model to an exported program using torch.export
    2. Inlines the exported program into a graph module

    Args:
        model (PreTrainedModel | PeftModel): The PyTorch model to export
        hints (dict[str, TensorTypeHint]): Configuration for input arguments
        enable_experimental_decompositions (bool, optional): Whether to enable experimental decompositions.
            Defaults to False.

    Returns:
        GraphModule: The inlined graph module
    """
    logger.info("Running torch.export")
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


def build_multimodal_engine_components(
    graph_module: GraphModule,
    hints: dict[str, TensorTypeHint],
    *,
    output_names: list[str],
    builder_config: TRTMultiModalBuildConfig,
    trt_config: TensorRTConfig | None = None,
    engine_cache: BaseEngineCache | None = None,
    network_name: str | None = None,
    debug_node_names: list[str] | None = None,
) -> Generator[tuple[str, EngineComponent], None, None]:
    """Build TensorRT engine components for a multimodal model.

    Args:
        graph_module (GraphModule): The graph module to convert to TensorRT engines
        hints (dict[str, TensorTypeHint]): Configuration for input arguments
        output_names (list[str]): List of output names
        builder_config (TRTMultiModalBuildConfig): Configuration for building TensorRT engines
        trt_config (TensorRTConfig | None): TensorRT builder configuration. Defaults to None.
        engine_cache (BaseEngineCache | None): Cache for storing/loading built engines. Defaults to None.
        network_name (str | None): Name of the network. Defaults to None.
        debug_node_names (list[str] | None): List of node names to output for debugging. Defaults to None.
    """
    logger.opt(lazy=True).debug("Memory Footprint: {m}", m=get_memory_footprint)
    save_for_debug("vl_graph_module_rank0", graph_module)

    logger.info("Building TensorRT engine")
    if debug_node_names:
        output_names.extend(debug_node_names)

    serialized_engine = convert(
        graph_module,
        hints,
        trt_config or TensorRTConfig(),
        output_names=output_names,
        engine_cache=engine_cache,
        network_name=network_name,
    )
    yield ("model.engine", serialized_engine)
    yield ("config.json", builder_config.as_dict())


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
        "Building TensorRT-LLM engine{for_rank}",
        for_rank=f" for rank {rank}" if argument_hint.mapping.world_size > 1 else "",
    )

    output_names = ["logits" if argument_hint.mapping.is_last_pp_rank() else "hidden_states_output"]
    if debug_node_names:
        output_names.extend(debug_node_names)
    yield (
        f"rank{rank}.engine",
        convert(
            graph_module,
            argument_hint,
            trt_config or TensorRTConfig(),
            output_names=output_names,
            engine_cache=engine_cache,
            network_name=network_name,
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
        exclude = {"pretrained_config": {"moe"}} if component.pretrained_config.moe is None else None
        with open(component_path, "w") as f:
            f.write(component.model_dump_json(indent=2, exclude=exclude))
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
    elif isinstance(component, dict):
        logger.info(f"Writing config at {component_path}")
        with open(component_path, "w") as f:
            json.dump(component, f, indent=2)
    else:
        raise ValueError(f"Unsupported component type: {type(component)}")


def add_outputs(names: list[str]) -> Callable[[GraphModule, CompilationSettings], GraphModule]:
    """Create a transform pass that adds additional outputs to the graph module.

    Args:
        names (list[str]): List of node names to add as additional outputs

    Returns:
        Callable[[GraphModule], GraphModule]: A callable that transforms a graph module by adding the specified nodes
            as outputs
    """

    # pylint: disable-next=unused-argument
    def reset_output(gm: GraphModule, compilation_settings: CompilationSettings) -> GraphModule:
        """Add specified nodes as additional outputs to the graph module.

        Args:
            gm (GraphModule): The graph module to modify
            compilation_settings (CompilationSettings): The compilation settings

        Returns:
            GraphModule: The modified graph module with additional outputs

        Raises:
            RuntimeError: If output node is not found or specified nodes don't exist
        """
        nodes = {n.name: n for n in gm.graph.nodes}
        node = find_output_node(gm)

        try:
            outputs = node.args[0] + tuple(nodes[name] for name in names)  # type: ignore
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
