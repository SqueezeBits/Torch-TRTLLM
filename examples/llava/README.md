# Building LLaVA Models with Ditto

This guide demonstrates how to build TensorRT engines for LLaVA (Large Language and Vision Assistant) models using the **Ditto**. LLaVA is a multimodal model that combines vision understanding with language generation capabilities.

## Overview

LLaVA models consist of two main components:
1. **Vision Encoder**: Processes input images to extract visual features
2. **Language Model**: Generates text responses based on visual and textual inputs

Ditto builds separate TensorRT engines for each component like TensorRT-LLM does.

## Quick Start

### 1. Basic LLaVA Model Building

The example script `build_llava.py` demonstrates how to build TensorRT engines for a LLaVA model:

```python
import torch
from transformers import LlavaForConditionalGeneration
from ditto.api import build_llm_engine, build_multimodal_engine
from ditto.arguments import DynamicDimension, TensorTypeHint
from ditto.configs import TensorRTBuilderConfig, TensorRTConfig, TensorRTNetworkCreationFlags

# Configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OUTPUT_DIR = "engines/llava-1.5-7b-hf"
MAX_BATCH_SIZE = 4
VISION_FEATURE_SIZE = 576
```

### 2. Vision Encoder Wrapper

Create a wrapper for the vision encoder to extract the specific features needed. The wrapper simplifies the original HuggingFace vision encoder interface which expects additional inputs beyond just the image tensor and combines the separated vision modules (vision_tower and project) into a single module.

```python
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
```

### 3. Load the Model

```python
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
vision_wrapper = LlavaVisionWrapper(model)
```

### 4. Build Language Model Engine

```python
build_llm_engine(
    model.language_model,
    f"{OUTPUT_DIR}/llm",
    max_batch_size=MAX_BATCH_SIZE,
    max_prompt_embedding_table_size=VISION_FEATURE_SIZE * MAX_BATCH_SIZE,
)
```

### 5. Build Vision Encoder Engine

```python
input_spec = TensorTypeHint(
    shape=(
        DynamicDimension(name="vl_batch_size", min=1, opt=max(1, MAX_BATCH_SIZE // 2), max=MAX_BATCH_SIZE),
        3,  # RGB channels
        336,  # Height
        336,  # Width
    ),
    dtype=vision_wrapper.dtype,
)

build_multimodal_engine(
    vision_wrapper,
    f"{OUTPUT_DIR}/vision",
    max_batch_size=MAX_BATCH_SIZE,
    input_specs=[input_spec],
    input_names=["input"],
    output_names=["encoder_output"],
    model_type="llava",
    trt_config=TensorRTConfig(
        network_creation_flags=TensorRTNetworkCreationFlags(strongly_typed=False),
        builder_config=TensorRTBuilderConfig(fp16=True),
    ),
)
```

### 6. Run LLaVA Engine

To run the engine you've built, use [run.py](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.19.0/examples/multimodal/run.py) from TensorRT-LLM repository.

Here's an example command:
```shell
$ python tensorrt_llm/examples/multimodal/run.py \
    --batch_size 2 \
    --engine_dir ./engines/llava-1.5-7b-hf \
    --hf_model_dir llava-hf/llava-1.5-7b-hf \
    --batch_size 2 \
    --input_text "Describe this image. Answer:" "Describe this image. Answer" \
    --run_profiling \
    --session cpp
```

After running the command, you'll see output similar to this:
```
...
[06/24/2025-16:22:40] [TRT-LLM] [I] ---------------------------------------------------------
[06/24/2025-16:22:40] [TRT-LLM] [I] 
[Q] ['Describe this image. Answer:', 'Describe this image. Answer']
[06/24/2025-16:22:40] [TRT-LLM] [I] 
[A]: ['The image features a large fountain with a water jet shooting high into the air, located near a body of water. The fountain is situated in front of a building, which appears to be a hotel or a similar establishment. The water jet is spraying water high into the air, creating a visually appealing scene.\n\nIn the vicinity, there are several people enjoying the view and the atmosphere. Some of them are standing closer to the fountain, while others are further away, taking in the entire scene. The presence of the water fountain and the people adds a lively and']
[06/24/2025-16:22:40] [TRT-LLM] [I] 
[A]: ["The image features a large fountain with a water jet shooting high into the air, located near a body of water. The fountain is situated in front of a building, which appears to be a hotel or a similar establishment. The water jet is spraying water high into the air, creating a visually appealing scene.\n\nThere are several people in the vicinity of the fountain, enjoying the view and the atmosphere. Some of them are standing closer to the fountain, while others are further away, taking in the entire scene. The presence of the people and the fountain'"]
[06/24/2025-16:26:36] [TRT-LLM] [I] Generated 128 tokens
[06/24/2025-16:26:36] [TRT-LLM] [I] Latencies per batch (msec)
[06/24/2025-16:26:36] [TRT-LLM] [I] e2e generation: 2961.5
[06/24/2025-16:26:36] [TRT-LLM] [I]   Preprocessing: 1.3
[06/24/2025-16:26:36] [TRT-LLM] [I]     Vision encoder: 0.2
[06/24/2025-16:26:36] [TRT-LLM] [I]   LLM generate: 2915.1
[06/24/2025-16:26:36] [TRT-LLM] [I]   Tokenizer decode: 45.1
[06/24/2025-16:22:40] [TRT-LLM] [I] ---------------------------------------------------------
```
