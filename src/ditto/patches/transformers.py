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

import torch
import transformers
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from .patch import custom_patch


@custom_patch(
    name="transformers.modeling_attn_mask_utils.AttentionMaskConverter._make_causal_mask",
    reason=(
        "resolving torch.export failure observed in transformers<4.48.0 by some models. "
        "See https://github.com/huggingface/transformers/pull/35291, "
        "which is included in the transformers-4.48.0 release."
    ),
    required=transformers.__version__ < "4.48.0",
    env_var_to_disable="DISABLE_TRANSFORMERS_ATTENTION_MASK_CONVERTER_PATCH",
)
def patch_attention_mask_converter_make_causal_mask() -> None:
    def patched_attention_mask_converter_make_causal_mask(
        cls: type[AttentionMaskConverter],
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        """Make causal mask used for bi-directional self-attention.

        Adapted for resolving torch.export failure.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            # Recent changes in PyTorch prevent mutations on tensors converted with aten::_to_copy
            # See https://github.com/pytorch/pytorch/issues/127571
            if torch._dynamo.is_compiling():
                mask = mask.clone()
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    AttentionMaskConverter._make_causal_mask = patched_attention_mask_converter_make_causal_mask
