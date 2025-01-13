from pydantic import TypeAdapter, ValidationError
from tensorrt_llm._utils import trt_dtype_to_str
from torch.fx import Graph, GraphModule
from transformers import PretrainedConfig

from .configs import (
    DTypeLiteral,
    TRTLLMBuildConfig,
    TRTLLMEngineConfig,
    TRTLLMMapping,
    TRTLLMPretrainedConfig,
)
from .fx.subgraphs import TokenEmbedding
from .fx.targets import GPTAttentionPlugin


class PretrainedConfigGenerationError(RuntimeError):
    """Error indicating failure in pretrained config generation based on graph module."""


def generate_trtllm_engine_config(
    graph_module: GraphModule,
    build_config: TRTLLMBuildConfig,
    mapping: TRTLLMMapping,
    *,
    architecture: str | None = None,
) -> TRTLLMEngineConfig:
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
    vocab_size, hidden_size = get_embedding_weight_sizes(graph_module)
    pretrained_config = generate_and_validate_configs(
        collect_gpt_attention_plugins(graph_module.graph),
        architecture=architecture or "UnknownLanguageModel",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        mapping=mapping,
    )
    if "qwen" in pretrained_config.architecture.lower() and isinstance(
        hf_config := graph_module.meta.get("pretrained_config"), PretrainedConfig
    ):
        # pylint: disable-next=unsupported-assignment-operation
        pretrained_config.extra_fields["qwen_type"] = hf_config.model_type
    return pretrained_config


def get_embedding_weight_sizes(graph_module: GraphModule) -> tuple[int, int]:
    for node in graph_module.graph.nodes:
        if token_embedding := TokenEmbedding.configure_from(node):
            return token_embedding.vocab_size, token_embedding.hidden_size
    raise PretrainedConfigGenerationError("Failed to infer vocab size from graph module")


def collect_gpt_attention_plugins(graph: Graph) -> list[GPTAttentionPlugin]:
    plugins: list[GPTAttentionPlugin] = []
    for node in graph.nodes:
        if node.op == "call_function" and isinstance(node.target, GPTAttentionPlugin):
            plugins.append(node.target)
    return sorted(plugins, key=lambda plugin: plugin.layer_idx)


def generate_and_validate_configs(
    plugins: list[GPTAttentionPlugin],
    *,
    vocab_size: int,
    hidden_size: int,
    architecture: str,
    mapping: TRTLLMMapping,
) -> TRTLLMPretrainedConfig:
    if (num_hidden_layers := len(plugins)) == 0:
        raise PretrainedConfigGenerationError("No GPTAttentionPlugin nodes found")

    first_plugin, *other_plugins = plugins
    config = generate_config(
        0,
        first_plugin,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        architecture=architecture,
        num_hidden_layers=num_hidden_layers,
        mapping=mapping,
    )

    for i, other_plugin in enumerate(other_plugins):
        plugin_idx = i + 1  # the first (index=0) plugin has been popped out
        other_config = generate_config(
            plugin_idx,
            other_plugin,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            architecture=architecture,
            num_hidden_layers=num_hidden_layers,
            mapping=mapping,
        )
        if config != other_config:
            raise PretrainedConfigGenerationError(
                f"pretrained config mismatched at {plugin_idx}-th GPTAttentionPlugin node:\n"
                f"  * config at index 0: {repr(config)}\n"
                f"  * config at index {plugin_idx}: {repr(other_config)}"
            )
    return config


def generate_config(
    plugin_idx: int,
    plugin: GPTAttentionPlugin,
    *,
    vocab_size: int,
    hidden_size: int,
    architecture: str,
    num_hidden_layers: int,
    mapping: TRTLLMMapping,
) -> TRTLLMPretrainedConfig:
    if plugin_idx != plugin.layer_idx:
        raise PretrainedConfigGenerationError(f"Expected layer_idx={plugin_idx} but got {plugin.layer_idx=}")

    try:
        dtype: DTypeLiteral = TypeAdapter(DTypeLiteral).validate_python(trt_dtype_to_str(plugin.type_id))
    except (ValidationError, KeyError) as e:
        raise PretrainedConfigGenerationError(
            f"Found GPT attention plugin with invalid type_id={plugin.type_id}."
        ) from e

    return TRTLLMPretrainedConfig(
        architecture=architecture,
        dtype=dtype,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=plugin.num_heads,
        num_key_value_heads=plugin.num_kv_heads,
        mapping=mapping,
        # TODO: fill in appropriate values in the quantization when quantization are supported.
        # quantization=TRTLLMQuantConfig(...),
    )
