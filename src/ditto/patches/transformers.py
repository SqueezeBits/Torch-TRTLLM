from loguru import logger
import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def patched_attention_mask_converter_make_causal_mask(
    cls: AttentionMaskConverter,
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

logger.info("ditto patches for transformers are applied!")
