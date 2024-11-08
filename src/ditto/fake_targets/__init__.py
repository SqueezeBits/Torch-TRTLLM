from .gpt_attention_plugin import (
    FakeGPTAttentionPlugin,
    GPTAttentionPluginInputs,
    Llama3ScalingConfig,
    ROPEConfig,
)
from .rope import (
    FAKE_ROPE_TARGETS,
    fake_llama2_rope,
    get_llama2_rope_pattern_graph,
    get_llama2_rope_replacment_graph,
)
from .transposed_mm import fake_transposed_mm
