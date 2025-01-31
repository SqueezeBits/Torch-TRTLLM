from tensorrt_llm._utils import trt_dtype_to_str
from torch.fx import Graph, GraphModule
from transformers import PretrainedConfig

from .configs import (
    TRTLLMBuildConfig,
    TRTLLMEngineConfig,
    TRTLLMLoraConfig,
    TRTLLMMapping,
    TRTLLMPretrainedConfig,
)
from .fx.subgraphs import Linear, TokenEmbedding
from .fx.targets import GPTAttentionPlugin
from .literals import DTypeLiteral
from .types import verify


class PretrainedConfigGenerationError(RuntimeError):
    """Indicate failure in pretrained config generation based on graph module."""


def generate_trtllm_engine_config(
    graph_module: GraphModule,
    build_config: TRTLLMBuildConfig,
    mapping: TRTLLMMapping,
    *,
    architecture: str | None = None,
) -> TRTLLMEngineConfig:
    """Generate TRTLLM engine configuration.

    Args:
        graph_module (GraphModule): The graph module to process.
        build_config (TRTLLMBuildConfig): The build configuration.
        mapping (TRTLLMMapping): The mapping configuration.
        architecture (str | None): The architecture name, optional.

    Returns:
        TRTLLMEngineConfig: The generated engine configuration.
    """
    if (lora_config := graph_module.meta.pop("lora_config", None)) is not None:
        build_config.lora_config = TRTLLMLoraConfig.model_validate(lora_config)
        build_config.plugin_config.lora_plugin = "auto"
    return TRTLLMEngineConfig(
        pretrained_config=generate_trtllm_pretrained_config(
            graph_module,
            mapping,
            architecture=architecture,
        ),
        build_config=build_config,
    )


def generate_trtllm_pretrained_config(
    graph_module: GraphModule,
    mapping: TRTLLMMapping,
    *,
    architecture: str | None = None,
) -> TRTLLMPretrainedConfig:
    """Generate TRTLLM pretrained configuration.

    Args:
        graph_module (GraphModule): The graph module to process.
        mapping (TRTLLMMapping): The mapping configuration.
        architecture (str | None): The architecture name, optional.

    Returns:
        TRTLLMPretrainedConfig: The generated pretrained configuration.
    """
    vocab_size, hidden_size = get_embedding_weight_sizes(graph_module)
    pretrained_config = infer_and_validate_pretrained_configs(
        collect_gpt_attention_plugins(graph_module.graph),
        architecture=architecture or "UnknownLanguageModel",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=get_intermediate_size(graph_module),
        mapping=mapping,
    )
    if "qwen" in pretrained_config.architecture.lower() and isinstance(
        hf_config := graph_module.meta.get("pretrained_config"), PretrainedConfig
    ):
        # pylint: disable-next=unsupported-assignment-operation
        pretrained_config.extra_fields["qwen_type"] = hf_config.model_type
    return pretrained_config


def get_embedding_weight_sizes(graph_module: GraphModule) -> tuple[int, int]:
    """Get embedding weight sizes from the graph module.

    Args:
        graph_module (GraphModule): The graph module to process.

    Returns:
        tuple[int, int]: The vocabulary size and hidden size.

    Raises:
        PretrainedConfigGenerationError: If vocab size cannot be inferred.
    """
    for node in graph_module.graph.nodes:
        if token_embedding := TokenEmbedding.configure_from(node):
            return token_embedding.vocab_size, token_embedding.hidden_size
    raise PretrainedConfigGenerationError("Failed to infer vocab size from graph module")


def collect_gpt_attention_plugins(graph: Graph) -> list[GPTAttentionPlugin]:
    """Collect GPT attention plugins from the graph.

    Args:
        graph (Graph): The graph to process.

    Returns:
        list[GPTAttentionPlugin]: Sorted list of GPT attention plugins.
    """
    plugins: list[GPTAttentionPlugin] = []
    for node in graph.nodes:
        if node.op == "call_function" and isinstance(node.target, GPTAttentionPlugin):
            plugins.append(node.target)
    return sorted(plugins, key=lambda plugin: plugin.layer_idx)


def get_intermediate_size(graph_module: GraphModule) -> int:
    """Get intermediate size from the graph module.

    Args:
        graph_module (GraphModule): The graph module to process.

    Returns:
        int: The intermediate size.

    Raises:
        PretrainedConfigGenerationError: If no or multiple intermediate sizes are found.
    """
    values: set[int] = set()
    for node in graph_module.graph.nodes:
        if (linear := Linear.configure_from(node)) and linear.lora_prefix == "mlp_4h_to_h":
            values.add(linear.in_features)

    if len(values) == 0:
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
    """Infer and validate pretrained configurations.

    Args:
        plugins (list[GPTAttentionPlugin]): List of GPT attention plugins.
        vocab_size (int): The vocabulary size.
        hidden_size (int): The hidden size.
        architecture (str): The architecture name.
        intermediate_size (int): The intermediate size.
        mapping (TRTLLMMapping): The mapping configuration.

    Returns:
        TRTLLMPretrainedConfig: The validated pretrained configuration.

    Raises:
        PretrainedConfigGenerationError: If no plugins are found or configurations mismatch.
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
    """Infer pretrained configuration for a plugin.

    Args:
        plugin_idx (int): The index of the plugin.
        plugin (GPTAttentionPlugin): The GPT attention plugin.
        vocab_size (int): The vocabulary size.
        hidden_size (int): The hidden size.
        architecture (str): The architecture name.
        num_hidden_layers (int): The number of hidden layers.
        intermediate_size (int): The intermediate size.
        mapping (TRTLLMMapping): The mapping configuration.

    Returns:
        TRTLLMPretrainedConfig: The inferred pretrained configuration.

    Raises:
        PretrainedConfigGenerationError: If layer index mismatches or invalid type_id is found.
    """
    if plugin_idx != plugin.layer_idx:
        raise PretrainedConfigGenerationError(f"Expected layer_idx={plugin_idx} but got {plugin.layer_idx=}")

    if (dtype := verify(trt_dtype_to_str(plugin.type_id), as_type=DTypeLiteral)) is None:
        raise PretrainedConfigGenerationError(f"Found GPT attention plugin with invalid type_id={plugin.type_id}.")

    return TRTLLMPretrainedConfig(
        architecture=architecture,
        dtype=dtype,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=plugin.num_heads,
        num_key_value_heads=plugin.num_kv_heads,
        intermediate_size=intermediate_size,
        mapping=mapping,
        # TODO: fill in appropriate values in the quantization when quantization are supported.
        # quantization=TRTLLMQuantConfig(...),
    )
