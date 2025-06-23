import math

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
)

from ditto.api import build_llm_engine, build_multimodal_engine
from ditto.arguments import DynamicDimension, TensorTypeHint

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "engines/qwen2.5-vl"
DTYPE = torch.float16
IMAGE_SIZE = (504, 504)
MAX_BATCH_SIZE = 4
VISION_FEATURE_SIZE = 324  # for image size (504, 504)

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


# Based on tensorrt_llm/tools/multimodal_builder.py
class VisionAttentionOpt(Qwen2_5_VLVisionAttention):
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__(dim, num_heads)
        self.head_dim = dim / num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        # Copied from transformers.models.llama.modeling_qwen2_vl
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb_vision(
            q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            orig_q_dtype = q.dtype
            orig_k_dtype = k.dtype
            q, k = q.float(), k.float()
            cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            q_embed = q_embed.to(orig_q_dtype)
            k_embed = k_embed.to(orig_k_dtype)
            return q_embed, k_embed

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5VLVisionBlockOpt(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "eager") -> None:
        super().__init__(config)
        self.attn = VisionAttentionOpt(config.hidden_size, num_heads=config.num_heads)

    def forward(self, hidden_states, position_embeddings, attention_mask) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5VisionTransformerPretrainedModelOpt(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, image_size: tuple[int, int]) -> None:
        super().__init__(config)
        self.blocks = torch.nn.ModuleList([Qwen2_5VLVisionBlockOpt(config) for _ in range(config.depth)])
        self.height, self.width = image_size[0], image_size[1]
        self.patch_size = 14
        self.merge_size = 2
        self.temporal_patch_size = 2

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # assume that resize, rescale, normalize, convert rgb are already done
        # image.shape == (bs, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        batch_size = images.shape[0]
        patches = images.repeat_interleave(2, dim=0)
        channel = patches.shape[1]
        grid_h, grid_w = self.height // self.patch_size, self.width // self.patch_size
        patches = patches.reshape(batch_size, self.temporal_patch_size, channel, -1)
        patches = patches.permute(0, 3, 2, 1)
        patches = patches.reshape(
            batch_size,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
            -1,
        )
        patches = patches.permute(0, 1, 4, 2, 5, 7, 3, 6)
        flatten_patches = patches.reshape(
            batch_size * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return flatten_patches

    def forward(
        self,
        images: torch.Tensor,
        window_index: torch.Tensor,
        reverse_indices: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_mask_window: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.preprocess(images)
        hidden_states = self.patch_embed(hidden_states)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        # hidden_states = hidden_states[window_index, :, :]
        hidden_states = torch.nn.functional.embedding(window_index, hidden_states.view(hidden_states.shape[0], -1))
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        # rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = torch.nn.functional.embedding(window_index, rotary_pos_emb.view(rotary_pos_emb.shape[0], -1))
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask = attention_mask
            else:
                attention_mask = attention_mask_window
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        hidden_states = self.merger(hidden_states)
        # reverse_indices = torch.argsort(window_index)
        # hidden_states = hidden_states[reverse_indices, :]
        hidden_states = torch.nn.functional.embedding(reverse_indices, hidden_states)
        return hidden_states


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

        image_grid_thw = torch.tensor(
            [[1, image_size[0] // self.visual.patch_size, image_size[1] // self.visual.patch_size]]
            * (max_batch_size + 1)
        )
        window_index, cu_window_seqlens = self.visual.get_window_index(image_grid_thw)

        self.seqlen_per_batch = (image_size[0] // self.visual.patch_size) * (image_size[1] // self.visual.patch_size)
        self.vision_feature_seqlen_per_batch = self.seqlen_per_batch // self.visual.spatial_merge_unit
        self.window_index = torch.tensor(window_index, device=self.device)
        self.rotary_pos_emb = self.visual.rot_pos_emb(image_grid_thw)
        self.cu_seqlens = torch.nn.functional.pad(
            torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(dim=0),
            (1, 0),
            value=0,
        )
        self.cu_window_seqlens = torch.unique_consecutive(torch.tensor(cu_window_seqlens, device=self.device))
        self.reverse_indices = torch.argsort(self.window_index)

        attention_mask = torch.zeros(
            [1, self.cu_seqlens[-1], self.cu_seqlens[-1]], device=self.device, dtype=torch.bool
        )
        for i in range(1, len(self.cu_seqlens)):
            attention_mask[
                ..., self.cu_seqlens[i - 1] : self.cu_seqlens[i], self.cu_seqlens[i - 1] : self.cu_seqlens[i]
            ] = True
        self.attention_mask = attention_mask

        attention_mask = torch.zeros(
            [1, self.cu_seqlens[-1], self.cu_seqlens[-1]], device=self.device, dtype=torch.bool
        )
        for i in range(1, len(self.cu_window_seqlens)):
            attention_mask[
                ...,
                self.cu_window_seqlens[i - 1] : self.cu_window_seqlens[i],
                self.cu_window_seqlens[i - 1] : self.cu_window_seqlens[i],
            ] = True
        self.attention_mask_window = attention_mask

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

if __name__ == "__main__":
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    vision_wrapper = Qwen2_5VLVisionWrapper(model, IMAGE_SIZE, MAX_BATCH_SIZE)

    # build language model
    build_llm_engine(
        Qwen2_5VLLLMWrapper(model),
        f"{OUTPUT_DIR}/llm",
        network_name=Qwen2_5_VLForConditionalGeneration.__name__,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=8192,
        max_prompt_embedding_table_size=MAX_BATCH_SIZE * VISION_FEATURE_SIZE,
    )

    # build vision encoder
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
