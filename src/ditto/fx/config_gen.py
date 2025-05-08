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

from tensorrt_llm._utils import trt_dtype_to_str
from torch.fx import Graph, GraphModule
from transformers import PretrainedConfig

from ..configs import (
    TRTLLMBuildConfig,
    TRTLLMEngineConfig,
    TRTLLMLoraConfig,
    TRTLLMMapping,
    TRTLLMMoEConfig,
    TRTLLMPretrainedConfig,
    TRTLLMQuantConfig,
)
from ..literals import DTypeLiteral
from ..quantization import GlobalQuantConfig
from ..types import DataType, verify
from .metadata_keys import LINEAR_TYPE, MOE_CONFIG
from .nodes import Fp8RowwiseGemm, RmsnormQuantization, WeightOnlyGroupwiseQuantMatmul, WeightOnlyQuantMatmul
from .subgraphs import Linear, TokenEmbedding
from .targets import GPTAttentionPlugin, MixtureOfExpertsPlugin


class PretrainedConfigGenerationError(RuntimeError):
    """Error indicating failure in pretrained config generation based on graph module."""


def generate_trtllm_engine_config(
    graph_module: GraphModule,
    build_config: TRTLLMBuildConfig,
    mapping: TRTLLMMapping,
    *,
    global_quant_config: GlobalQuantConfig | None = None,
    architecture: str | None = None,
) -> TRTLLMEngineConfig:
    """Generate TRTLLM engine configuration.

    Args:
        graph_module (GraphModule): The graph module to process.
        build_config (TRTLLMBuildConfig): The build configuration.
        mapping (TRTLLMMapping): The mapping configuration.
        global_quant_config (GlobalQuantConfig | None): The global quantization configuration. Defaults to None.
        architecture (str | None): The architecture name, optional. Defaults to None.

    Returns:
        TRTLLMEngineConfig: The generated engine configuration.
    """
    if (lora_config := graph_module.meta.pop("lora_config", None)) is not None:
        build_config.lora_config = TRTLLMLoraConfig.model_validate(lora_config)
        build_config.plugin_config.lora_plugin = "auto"
    if graph_module.meta.get(MOE_CONFIG, None) is not None:
        build_config.plugin_config.moe_plugin = "auto"

    build_config.plugin_config.fp8_rowwise_gemm_plugin = (
        DataType(fp8_rowwise_gemm.target.type_id).to(str)  # type: ignore[assignment]
        if (fp8_rowwise_gemm := Fp8RowwiseGemm.find_in(graph_module.graph))
        else None
    )
    build_config.plugin_config.weight_only_groupwise_quant_matmul_plugin = (
        DataType(woq_group_mm.target.type_id).to(str)  # type: ignore[assignment]
        if (woq_group_mm := WeightOnlyGroupwiseQuantMatmul.find_in(graph_module.graph))
        else None
    )
    build_config.plugin_config.weight_only_quant_matmul_plugin = (
        DataType(woq_mm.target.type_id).to(str)  # type: ignore[assignment]
        if (woq_mm := WeightOnlyQuantMatmul.find_in(graph_module.graph))
        else None
    )
    build_config.plugin_config.rmsnorm_quantization_plugin = (
        DataType(rms_quant.target.type_id).to(str)  # type: ignore[assignment]
        if (rms_quant := RmsnormQuantization.find_in(graph_module.graph))
        else None
    )
    build_config.plugin_config.quantize_per_token_plugin = build_config.plugin_config.quantize_tensor_plugin = (
        build_config.plugin_config.fp8_rowwise_gemm_plugin is not None
    )

    return TRTLLMEngineConfig(
        pretrained_config=generate_trtllm_pretrained_config(
            graph_module,
            mapping,
            global_quant_config=global_quant_config,
            architecture=architecture,
        ),
        build_config=build_config,
    )


def generate_trtllm_pretrained_config(
    graph_module: GraphModule,
    mapping: TRTLLMMapping,
    *,
    global_quant_config: GlobalQuantConfig | None = None,
    architecture: str | None = None,
) -> TRTLLMPretrainedConfig:
    """Generate TRTLLMPretrainedConfig from graph module.

    Args:
        graph_module (GraphModule): The graph module to generate the pretrained config from.
        mapping (TRTLLMMapping): The tensor parallel mapping to use for the pretrained config.
        global_quant_config (GlobalQuantConfig | None): The global quantization configuration. Defaults to None.
        architecture (str | None, optional): The architecture to use for the pretrained config. Defaults to None.

    Returns:
        TRTLLMPretrainedConfig: The generated pretrained config.
    """
    vocab_size, hidden_size = get_embedding_weight_sizes(graph_module)
    pretrained_config = infer_and_validate_pretrained_configs(
        collect_gpt_attention_plugins(graph_module.graph),
        architecture=architecture or "UnknownLanguageModel",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=get_intermediate_size(graph_module, mapping),
        mapping=mapping,
    )
    pretrained_config.quantization = TRTLLMQuantConfig.create_from(global_quant_config) if global_quant_config else None
    if isinstance(hf_config := graph_module.meta.get("pretrained_config"), PretrainedConfig):
        if "qwen" in pretrained_config.architecture.lower():
            pretrained_config.extra_fields["qwen_type"] = hf_config.model_type
        if "gemma2" in pretrained_config.architecture.lower():
            pretrained_config.extra_fields["query_pre_attn_scalar"] = hf_config.query_pre_attn_scalar
    if (moe_config := graph_module.meta.pop(MOE_CONFIG, None)) is not None:
        pretrained_config.moe = TRTLLMMoEConfig.model_validate(moe_config)
    return pretrained_config


def get_embedding_weight_sizes(graph_module: GraphModule) -> tuple[int, int]:
    """Get the vocab size and hidden size from the graph module.

    Args:
        graph_module (GraphModule): The graph module to get the vocab size and hidden size from.

    Returns:
        tuple[int, int]: The vocab size and hidden size.

    Raises:
        PretrainedConfigGenerationError: If failed to infer vocab size from graph module.
    """
    for node in graph_module.graph.nodes:
        if token_embedding := TokenEmbedding.configure_from(node):
            return token_embedding.vocab_size, token_embedding.hidden_size
    raise PretrainedConfigGenerationError("Failed to infer vocab size from graph module")


def collect_gpt_attention_plugins(graph: Graph) -> list[GPTAttentionPlugin]:
    """Collect GPTAttentionPlugin nodes from the graph.

    Args:
        graph (Graph): The graph to collect GPTAttentionPlugin nodes from.

    Returns:
        list[GPTAttentionPlugin]: The collected GPTAttentionPlugin nodes that are sorted by layer index.
    """
    plugins: list[GPTAttentionPlugin] = []
    for node in graph.nodes:
        if node.op == "call_function" and isinstance(node.target, GPTAttentionPlugin):
            plugins.append(node.target)
    return sorted(plugins, key=lambda plugin: plugin.layer_idx)


def get_intermediate_size(graph_module: GraphModule, mapping: TRTLLMMapping) -> int:
    """Get intermediate size from the graph module.

    Args:
        graph_module (GraphModule): The graph module to process
        mapping (TRTLLMMapping): The tensor parallel mapping configuration

    Returns:
        int: The intermediate size.

    Raises:
        PretrainedConfigGenerationError: If no or multiple intermediate sizes are found.
    """
    values: set[int] = set()
    values_of_shared_experts: set[int] = set()
    values_of_experts: set[int] = set()
    for node in graph_module.graph.nodes:
        if (linear := Linear.configure_from(node)) and linear.lora_prefix == "mlp_4h_to_h":
            if linear.mm.meta.get(LINEAR_TYPE) == "shared_expert":
                values_of_shared_experts.add(linear.in_features)
            else:
                values.add(linear.in_features)
        if isinstance(node.target, MixtureOfExpertsPlugin):
            values_of_experts.add(node.target.expert_inter_size * mapping.tp_size)

    if not (values := values or values_of_shared_experts or values_of_experts):
        raise PretrainedConfigGenerationError("No intermediate size found in graph module")

    if len(values) > 1:
        raise PretrainedConfigGenerationError(f"Found multiple intermediate sizes in graph module: {values}")

    return values.pop()


def infer_and_validate_pretrained_configs(
    plugins: list[GPTAttentionPlugin],
    *,
    vocab_size: int,
    hidden_size: int,
    architecture: str,
    intermediate_size: int,
    mapping: TRTLLMMapping,
) -> TRTLLMPretrainedConfig:
    """Infer and validate TRTLLMPretrainedConfig from GPTAttentionPlugin nodes.

    Args:
        plugins (list[GPTAttentionPlugin]): The GPTAttentionPlugin nodes to generate the pretrained config from.
        vocab_size (int): The vocab size to use for the pretrained config.
        hidden_size (int): The hidden size to use for the pretrained config.
        architecture (str): The architecture to use for the pretrained config.
        intermediate_size (int): The hidden size of the intermediate layer.
        mapping (TRTLLMMapping): The tensor parallel mapping to use for the pretrained config.

    Returns:
        TRTLLMPretrainedConfig: The generated pretrained config.

    Raises:
        PretrainedConfigGenerationError: If failed to generate the pretrained config.
    """
    if (num_hidden_layers := len(plugins)) == 0:
        raise PretrainedConfigGenerationError("No GPTAttentionPlugin nodes found")

    first_plugin, *other_plugins = plugins
    config = infer_pretrained_config(
        0,
        first_plugin,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        architecture=architecture,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        mapping=mapping,
    )

    for i, other_plugin in enumerate(other_plugins):
        plugin_idx = i + 1  # the first (index=0) plugin has been popped out
        other_config = infer_pretrained_config(
            plugin_idx,
            other_plugin,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            architecture=architecture,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            mapping=mapping,
        )
        if config != other_config:
            raise PretrainedConfigGenerationError(
                f"pretrained config mismatched at {plugin_idx}-th GPTAttentionPlugin node:\n"
                f"  * config at index 0: {repr(config)}\n"
                f"  * config at index {plugin_idx}: {repr(other_config)}"
            )
    return config


def infer_pretrained_config(
    plugin_idx: int,
    plugin: GPTAttentionPlugin,
    *,
    vocab_size: int,
    hidden_size: int,
    architecture: str,
    num_hidden_layers: int,
    intermediate_size: int,
    mapping: TRTLLMMapping,
) -> TRTLLMPretrainedConfig:
    """Infer TRTLLMPretrainedConfig from GPTAttentionPlugin node.

    Args:
        plugin_idx (int): The index of the GPTAttentionPlugin node.
        plugin (GPTAttentionPlugin): The GPTAttentionPlugin node to generate the pretrained config from.
        vocab_size (int): The vocab size to use for the pretrained config.
        hidden_size (int): The hidden size to use for the pretrained config.
        architecture (str): The architecture to use for the pretrained config.
        num_hidden_layers (int): The number of hidden layers to use for the pretrained config.
        intermediate_size (int): The hidden size of the intermediate layer.
        mapping (TRTLLMMapping): The tensor parallel mapping to use for the pretrained config.

    Returns:
        TRTLLMPretrainedConfig: The generated pretrained config.

    Raises:
        PretrainedConfigGenerationError: If failed to generate the pretrained config.
    """
    if plugin_idx != plugin.layer_idx:
        raise PretrainedConfigGenerationError(f"Expected layer_idx={plugin_idx} but got {plugin.layer_idx=}")

    if (dtype := verify(trt_dtype_to_str(plugin.type_id), as_type=DTypeLiteral)) is None:  # type: ignore[arg-type]
        raise PretrainedConfigGenerationError(f"Found GPT attention plugin with invalid type_id={plugin.type_id}.")

    return TRTLLMPretrainedConfig(
        architecture=architecture,
        dtype=dtype,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers * mapping.pp_size,
        num_attention_heads=plugin.num_heads * mapping.tp_size,
        num_key_value_heads=plugin.num_kv_heads * mapping.tp_size,
        intermediate_size=intermediate_size,
        mapping=mapping,
    )
