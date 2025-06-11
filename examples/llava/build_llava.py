import torch
from transformers import LlavaForConditionalGeneration

from ditto.api import build_llm_engine, build_multimodal_engine
from ditto.arguments import DynamicDimension, TensorTypeHint
from ditto.configs import TensorRTBuilderConfig, TensorRTConfig, TensorRTNetworkCreationFlags


class LlavaVisionWrapper(torch.nn.Module):
    def __init__(self, model: LlavaForConditionalGeneration):
        super().__init__()
        self.tower = model.vision_tower
        self.projector = model.multi_modal_projector
        self.feature_layer = model.config.vision_feature_layer
        self.dtype = self.tower.dtype
        self.device = self.tower.device
        self._supports_sdpa = model._supports_sdpa

    def forward(self, input):
        all_hidden_states = self.tower(input, output_hidden_states=True).hidden_states
        features = all_hidden_states[self.feature_layer][:, 1:]
        return self.projector(features)


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OUTPUT_DIR = "engines/llava-1.5-7b-hf"
max_batch_size = 4
vision_feature_size = 576

model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
vision_wrapper = LlavaVisionWrapper(model)

# build language model
build_llm_engine(
    model.language_model,
    f"{OUTPUT_DIR}/llm",
    max_batch_size=max_batch_size,
    max_prompt_embedding_table_size=vision_feature_size * max_batch_size,
)
del model.language_model

# build vision encoder
input_spec = TensorTypeHint(
    shape=(
        DynamicDimension(name="vl_batch_size", min=1, opt=max(1, max_batch_size // 2), max=max_batch_size),
        3,
        336,
        336,
    ),
    dtype=vision_wrapper.dtype,
)
build_multimodal_engine(
    vision_wrapper,
    f"{OUTPUT_DIR}/vision",
    max_batch_size=max_batch_size,
    input_specs=[input_spec],
    input_names=["input"],
    output_names=["encoder_output"],
    model_type="llava",
    trt_config=TensorRTConfig(
        network_creation_flags=TensorRTNetworkCreationFlags(strongly_typed=False),
        builder_config=TensorRTBuilderConfig(fp16=True),
    ),
)
