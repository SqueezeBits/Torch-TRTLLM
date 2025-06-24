# Building Qwen2.5-VL Models with Ditto

This guide demonstrates how to build TensorRT engines for Qwen2.5-VL (Qwen2.5 Vision Language) models using the **Ditto**. Qwen2.5-VL is a multimodal model that combines vision understanding with language generation capabilities, supporting high-resolution image processing.

## Overview

Qwen2.5-VL models consist of two main components:
1. **Vision Encoder**: Processes input images to extract visual features using a vision transformer with windowed attention
2. **Language Model**: Generates text responses based on visual and textual inputs

Ditto builds separate TensorRT engines for each component. TensorRT-LLM also builds each component, but TensorRT-LLM's vision encoder accepts image patches preprocessed using the Qwen method and additional inputs. For Qwen's vision language models, TensorRT-LLM runs the vision encoder in Python, not within the C++ session, because the C++ session cannot handle preprocessed image patches and additional inputs.

However, in this example, we've enabled the entire process to run within the C++ session. We achieved this by incorporating the preprocessing steps directly into the model's forward pass, allowing the vision encoder to execute as part of the C++ session.

## Quick Start

### 1. Basic Qwen2.5-VL Model Building

The example script `build_qwen2_5_vl_cpp_e2e.py` demonstrates how to build TensorRT engines for a Qwen2.5-VL model:

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ditto.api import build_llm_engine, build_multimodal_engine
from ditto.arguments import DynamicDimension, TensorTypeHint

# Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "engines/qwen2.5-vl"
DTYPE = torch.float16
IMAGE_SIZE = (504, 504)
MAX_BATCH_SIZE = 4
VISION_FEATURE_SIZE = 324  # for image size (504, 504)
```

### 2. Language Model Wrapper

Since the Qwen2.5-VL model's language model head (`lm_head`) is separate from its main language model, a dedicated wrapper is essential to provide a single module. This wrapper ensures that the `lm_head` is correctly integrated into the forward pass, allowing Ditto to processing the entire language model.

```python
class Qwen2_5VLLLMWrapper(torch.nn.Module):
    def __init__(self, model: Qwen2_5_VLForConditionalGeneration):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self.config = model.config
        self.dtype = model.dtype
        self.device = model.device
        self._supports_sdpa = model._supports_sdpa

    def forward(self, input_ids, use_cache):
        hidden_states = self.model(input_ids, use_cache=use_cache)
        logits = self.lm_head(hidden_states[0])
        return logits
```

### 3. Vision Encoder Wrapper

This wrapper for the vision encoder is designed to manage the Qwen2.5-VL model's windowed attention mechanism and image preprocessing. A key optimization implemented here is the pre-computation of various necessary inputs for the vision encoder. During the model's initialization, tensors such as `window_index`, `rotary_pos_emb`, and `attention_mask` are calculated as constants based on the `max_batch_size`. These pre-computed tensors are then directly used in the forward pass, significantly allowing Ditto to build the vision encoder into a single TensorRT engine.

**Note**: Due to a `torch.compile` guard bug, the pre-computed tensors are generated for `max_batch_size + 1` to prevent issues during compilation. This issue appears to be fixed in PyTorch 2.7+.


```python
class Qwen2_5VLVisionWrapper(torch.nn.Module):
    def __init__(self, model, image_size: tuple[int, int], max_batch_size: int = 1):
        super().__init__()
        self.visual = Qwen2_5VisionTransformerPretrainedModelOpt._from_config(
            model.config.vision_config,
            torch_dtype=DTYPE,
            image_size=image_size,
        ).to("cuda")
        self.visual.load_state_dict(model.visual.state_dict())
        self.dtype = model.visual.dtype
        self.device = model.visual.device
        self._supports_sdpa = model._supports_sdpa
        self.max_batch_size = max_batch_size
        
        # Pre-compute window indices and attention masks for efficiency
        image_grid_thw = torch.tensor(
            [[1, image_size[0] // self.visual.patch_size, image_size[1] // self.visual.patch_size]]
            * (max_batch_size + 1)
        )
        window_index, cu_window_seqlens = self.visual.get_window_index(image_grid_thw)
        
        # Store pre-computed tensors
        self.window_index = torch.tensor(window_index, device=self.device)
        self.rotary_pos_emb = self.visual.rot_pos_emb(image_grid_thw)
        # ... additional setup code

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input's shape == (batch_size, 3, 504, 504)
        batch_size = input.shape[0]
        img_features = self.visual(
            input,
            self.window_index[: batch_size * self.vision_feature_seqlen_per_batch],
            self.reverse_indices[: batch_size * self.vision_feature_seqlen_per_batch],
            self.rotary_pos_emb[: batch_size * self.seqlen_per_batch, :],
            self.attention_mask[:, : batch_size * self.seqlen_per_batch, : batch_size * self.seqlen_per_batch],
            self.attention_mask_window[:, : batch_size * self.seqlen_per_batch, : batch_size * self.seqlen_per_batch],
        )
        return img_features.view(batch_size, -1, img_features.shape[-1])
```

### 4. Load the Model

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=DTYPE, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
vision_wrapper = Qwen2_5VLVisionWrapper(model, IMAGE_SIZE, MAX_BATCH_SIZE)
```

### 5. Build Language Model Engine

```python
build_llm_engine(
    Qwen2_5VLLLMWrapper(model),
    f"{OUTPUT_DIR}/llm",
    network_name=Qwen2_5_VLForConditionalGeneration.__name__,
    max_batch_size=MAX_BATCH_SIZE,
    max_prompt_embedding_table_size=MAX_BATCH_SIZE * VISION_FEATURE_SIZE,
)
```

### 6. Build Vision Encoder Engine

```python
input_specs = [
    TensorTypeHint(
        shape=(
            DynamicDimension(name="vl_batch_size", min=1, opt=max(1, MAX_BATCH_SIZE // 2), max=MAX_BATCH_SIZE),
            3,
            504,
            504,
        ),
        dtype=vision_wrapper.dtype,
    ),
]

build_multimodal_engine(
    vision_wrapper,
    f"{OUTPUT_DIR}/vision",
    max_batch_size=MAX_BATCH_SIZE,
    input_specs=input_specs,
    input_names=["input"],
    output_names=["encoder_output"],
    model_type="qwen2_5_vl",
)
```

### 7. Run Qwen2.5-VL Engine

> You can encounter a `KeyError: 'Qwen2_5_VLForConditionalGeneration'` because TensorRT-LLM v0.19.0 does not support Qwen2.5-VL models. To fix this, you need to **update the** `MODEL_MAP` **dictionary** in the `tensorrt_llm/models/__init__.py` file. Add the following entry to the dictionary: `"Qwen2_5_VLForConditionalGeneration": QWenForCausalLM`.

To run the engine you've built, use the provided `run.py` script in this example directory:

```shell
$ python run.py \
    --engine-dir ./engines/qwen2.5-vl \
    --batch-size 2
```
You can use default images by omitting the `--images` and `--input-texts` arguments:

After running the command, you'll see output similar to this:
```
...
[Q] Question: Describe this image. Answer:
[A] ['The image depicts a surreal scene where two people appear to be walking on a body of water, possibly a river or a lake. The water is calm, and the sky is clear with a hint of sunset or sunrise. The people are wearing casual clothing, and one of them is holding a bag. The overall atmosphere is serene and somewhat dreamlike.']

[Q] Question: Describe this image. Answer:
[A] ['The image appears to be a composite of multiple photographs, likely taken from different angles or perspectives, creating a distorted and fragmented visual effect. The central focus seems to be a waterfront area with a dock and a building in the background. The sky is partly cloudy with a gradient of colors, suggesting either sunrise or sunset. There are several cylindrical objects, possibly trash cans, scattered on the ground in the foreground. The overall composition gives a surreal and abstract feel to the scene.']

---------------------------------------------------------
Latencies per batch (msec)
Load dataset: 725.8
e2e generation: 1470.6
  Preprocessing: 298.8
  Run model: 1149.5
Decode outputs: 22.3
---------------------------------------------------------
```