import torch
from tensorrt_llm.functional import RopeEmbeddingUtils
from torch.fx import GraphModule
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from ...fake_gpt_attention_plugin import FakeGPTAttentionPlugin, GPTAttentionPluginInputs, ROPEConfig


def populate_fake_gpt_attention_plugin_inputs(graph_module: GraphModule) -> GraphModule:
    graph = graph_module.graph
    fake_gpt_attention_plugin_kwargs: GPTAttentionPluginInputs | None = None
    for node in graph.nodes:
        if not (
            node.op == "call_function"
            and isinstance(fake_gpt_attention_plugin := node.target, FakeGPTAttentionPlugin)
            and not node.kwargs
            and node.all_input_nodes
            and isinstance(rope_config := node.meta.get("rope_config"), ROPEConfig)
        ):
            continue
        if fake_gpt_attention_plugin_kwargs is None:
            rotary_inv_freq, embed_positions = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                rope_config.rotary_embedding_max_positions,
                rope_config.rotary_embedding_dim,
                rope_config.rotary_embedding_base,
                rope_config.rotary_embedding_scale,
                rope_config.rotary_embedding_scale_type,
                {},
            )
            graph_module.register_parameter("rotary_inv_freq", torch.nn.Parameter(torch.from_numpy(rotary_inv_freq)))
            graph_module.register_parameter("rotary_cos_sin", torch.nn.Parameter(torch.from_numpy(embed_positions)))
            last_placeholder = [n for n in graph.nodes if n.op == "placeholder"][-1]
            with graph.inserting_after(last_placeholder):
                _ = graph.get_attr("rotary_cos_sin")
                _ = graph.get_attr("rotary_inv_freq")
            fake_gpt_attention_plugin_kwargs = GPTAttentionPluginInputs.find_from(graph)
        with graph.inserting_after(node):
            new_node = graph.call_function(
                fake_gpt_attention_plugin,
                (*node.args, *fake_gpt_attention_plugin_kwargs.model_dump().values()),
            )
        node.replace_all_uses_with(new_node)
    clean_up_graph_after_modifications(graph_module)
    return graph_module
