import json
import os
from typing import Any

from .debug.network import builder_config_as_dict
import numpy as np
import tensorrt as trt
import tensorrt_llm as trtllm
import tensorrt_llm.graph_rewriting as gw
import torch
from loguru import logger
from tensorrt_llm import Tensor, default_net, default_trtnet
from tensorrt_llm.functional import (
    AttentionMaskType,
    PositionEmbeddingType,
    QuantMode,
    RopeEmbeddingUtils,
    RotaryScalingType,
    _add_plugin_info,
    _create_tensor,
)
from tensorrt_llm.runtime.generation import GenerationSession
from torch_tensorrt.logging import TRT_LOGGER

from .debug import (
    EngineInfo,
    open_debug_artifact,
    save_onnx_without_weights,
)


def patched_trtllm_network_to_dot(self: trtllm.Network, path: str | None) -> str | None:
    with open_debug_artifact("trt_network_def.onnx", "wb") as f:
        if f:
            save_onnx_without_weights(EngineInfo.from_network_definition(self.trt_network).as_onnx(), f)
    return None


original_builder_build_engine = trtllm.Builder.build_engine


def patched_builder_build_engine(
    self: trtllm.Builder,
    network: trtllm.Network,
    builder_config: trtllm.BuilderConfig,
    managed_weights: dict[str, Any] | None = None,
) -> trt.IHostMemory:
    if layer_names_ := os.environ.get("TRTLLM_ADD_OUTPUT", None):
        try:
            layer_names: dict[str, str] = json.loads(layer_names_)
        except json.JSONDecodeError:
            logger.error(f"Invalid json provided to TRTLLM_ADD_OUTPUT: {layer_names_}")
            layer_names = {}
        if layer_names:
            net = network.trt_network
            for layer_idx in range(net.num_layers):
                layer = net.get_layer(layer_idx)
                if layer.name not in layer_names:
                    continue
                layer_alias = layer_names.pop(layer.name)
                for output_idx in range(layer.num_outputs):
                    output = layer.get_output(output_idx)
                    if layer.num_outputs > 1:
                        layer_alias = f"{layer_alias}_{output_idx}"
                    logger.info(f"Marking new output: {output.name} -> {layer_alias}")
                    net.mark_output(output)
                    output.name = layer_alias
            if layer_names:
                for layer_name in layer_names:
                    logger.error(f"No such layer found: {layer_name}")
                logger.info("The layer names are as follows:")
                print("\n".join(net.get_layer(layer_idx).name for layer_idx in range(net.num_layers)))
    serialized_engine = original_builder_build_engine(self, network, builder_config, managed_weights)
    with open_debug_artifact("builder_config.json") as f:
        if f:
            config_dict = builder_config_as_dict(builder_config.trt_builder_config)
            json.dump(config_dict, f, indent=2, sort_keys=True)
    with open_debug_artifact("trtllm_builder_config.json", "w") as f:
        if f:
            config_dict = builder_config.to_dict()
            json.dump(config_dict, f, indent=2, sort_keys=True)
    with (
        open_debug_artifact("trt_engine.onnx", "wb") as f,
        open_debug_artifact("trt_engine.json") as g,
    ):
        if f and g:
            with trt.Runtime(TRT_LOGGER) as runtime:
                engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(serialized_engine)
            inspector = engine.create_engine_inspector()
            engine_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            json.dump(engine_info, g, indent=2, sort_keys=True)
            save_onnx_without_weights(EngineInfo.model_validate(engine_info).as_onnx(), f)
    return serialized_engine


original_rope_embedding_utils_create_sinusoidal_positions_for_attention_plugin = (
    RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin
)


def patched_create_sinusoidal_positions_for_attention_plugin(
    num_pos: int,
    dim: int,
    theta: float = 10000.0,
    scale: float = 1.0,
    scale_type: RotaryScalingType = RotaryScalingType.none,
    rope_scaling_config: dict[str, Any] | None = None,
    dtype: type[np.number] = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    rotary_inv_freq, embed_positions = original_rope_embedding_utils_create_sinusoidal_positions_for_attention_plugin(
        num_pos, dim, theta, scale, scale_type, rope_scaling_config, dtype
    )
    with open_debug_artifact("rope_inputs.pt", "wb") as f:
        if f:
            torch.save(
                {
                    "rotary_inv_freq": torch.from_numpy(rotary_inv_freq),
                    "rotary_cos_sin": torch.from_numpy(embed_positions),
                },
                f,
            )
    return rotary_inv_freq, embed_positions


def patched_dump_debug_buffers(self: GenerationSession, step: int) -> None:
    debug_buffer = {**self.debug_buffer}
    if "host_kv_cache_pool_pointers" in debug_buffer:
        debug_buffer["kv_cache_pool"] = self.kv_cache_pool
    with open_debug_artifact(f"step{step}.pt", "wb") as f:
        if f:
            torch.save(debug_buffer, f)
    for name, value in debug_buffer.items():
        print(
            f"{name}: {value if (value.ndim < 3 or value.numel() < 100) else f'tensor with shape={(*value.shape,)}, dtype={value.dtype}'}"
        )


@gw.record_signature
def patched_gpt_attention(
    *,
    qkv: Tensor,
    past_key_value: Tensor,
    context_fmha_custom_mask: Tensor | None = None,
    sequence_length: Tensor,
    host_past_key_value_lengths: Tensor | None,
    host_max_attention_window_sizes: Tensor,
    host_sink_token_length: Tensor,
    context_lengths: Tensor | None,
    cache_indirection: Tensor | None,
    host_request_types: Tensor,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    hidden_size_per_head: int,
    q_scaling: float,
    qk_tanh_scale: float = 0.0,
    rotary_embedding_dim: int = 0,
    rotary_embedding_base: float = 10000.0,
    rotary_embedding_scale_type: RotaryScalingType = RotaryScalingType.none,
    rotary_embedding_short_m_scale: float = 1.0,
    rotary_embedding_long_m_scale: float = 1.0,
    rotary_embedding_scale: float = 1.0,
    rotary_embedding_max_positions: int = 1024,
    rotary_embedding_original_max_positions: int = 1024,
    position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.learned_absolute,
    rotary_inv_freq: Tensor | None = None,
    rotary_cos_sin: Tensor | None = None,
    kv_orig_quant_scale: Tensor | None = None,
    kv_quant_orig_scale: Tensor | None = None,
    attention_output_orig_quant_scale: Tensor | None = None,
    kv_cache_quant_mode: QuantMode = QuantMode(0),
    max_context_length: int | None = None,
    mask_type: AttentionMaskType = AttentionMaskType.causal,
    block_sparse_block_size: int = 64,
    block_sparse_homo_head_pattern: bool = False,
    block_sparse_num_local_blocks: int = 16,
    block_sparse_vertical_stride: int = 8,
    alibi_slopes: Tensor | None = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    vision_start: int = -1,
    vision_length: int = -1,
    kv_cache_block_offsets: Tensor | None = None,
    host_kv_cache_block_offsets: Tensor = None,
    host_kv_cache_pool_pointers: Tensor = None,
    do_cross_attention: bool = False,
    cross_qkv: Tensor | None = None,  # for cross attention
    cross_qkv_length: Tensor | None = None,  # for cross attention
    encoder_input_lengths: Tensor | None = None,  # for cross attention
    relative_attention_bias: Tensor | None = None,  # for relative attention
    max_distance: int = 0,  # for relative attention
    host_context_lengths: Tensor | None = None,  # for pad-free input mode
    qkv_bias: Tensor | None = None,
    use_cache: bool = True,
    spec_decoding_is_generation_length_variable: bool = False,
    spec_decoding_max_generation_length: int = 0,
    spec_decoding_generation_lengths: Tensor = None,
    spec_decoding_position_offsets: Tensor = None,
    spec_decoding_packed_mask: Tensor = None,
    host_runtime_perf_knobs: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Add an operation that performs the multi-head attention in GPT-like models.

    The signature of the function will change in the future release - we are in
    the process of simplifying the API. The current version is still
    work-in-progress! The following API is provided with hints regarding the
    arguments that are likely to be removed or merged with others in the future
    release.

    See docs/gpt_attention.md for the documentation of that function.

    Parameters:
        qkv: Tensor (On GPU)
            The input QKV tensor. Its shape is [batch_beam_size, max_seqlen, qkv_dim] in padded mode and [1, num_tokens, qkv_dim] in
            packed mode. Where qkv_dim depends on using MQA, GQA, or MHA. See QKV Input in docs/gpt_attention.md,

        past_key_value: Tensor (On GPU)
            The tensor that stores KV cache data. Its shape is
            [max_batch_size * max_beam_width, 2, num_kv_heads, max_seqlen, hidden_dim_per_head]
            in contiguous mode and
            [max_blocks, 2, num_kv_heads, num_tokens_per_block, hidden_dim_per_head]
            in paged mode. See KV Cache in docs/gpt_attention.md,

        context_fmha_custom_mask: Tensor (On GPU)
            The tensor that stores the packed custom mask for fmha.
            Its shape is [num_tokens, max_kv_seqlen / 32].

        sequence_lengths: Tensor (On GPU)
            The tensor that stores the length of each sequence. Its shape is
            [batch_size]. See QKV Input in docs/gpt_attention.md,

        host_past_key_value_lengths: Tensor (On CPU)
            An INT32 tensor of shape [batch_size],

        host_max_attention_window_sizes: Tensor (On CPU)
            An INT32 tensor of shape [1].
            by default, the max_attention_window_size is determined by the shape of cache_indir_table.
            And we support independent max_attention_window_size for each layer.
            This controls the sliding-window-attention/cyclic-kv-cache features.

        context_lengths: Tensor (On GPU)
            The tensor that stores the context-phase sequence length of each request. Its shape
            is [batch_size]. See QKV Input in doc/functional.py,

        cache_indirection: Tensor (On GPU)
            The tensor to reconstruct the paths when using beam-search. Its
            shape is [batch_size, beam_width, max_seqlen]. See Beam-Search in
            docs/gpt_attention.md,

        host_request_types: Tensor = None (On CPU)
            The tensor on the host that indicates if a request is in context or
            generation phase. Its shape is [batch_size]. See Inflight Batching
            in docs/gpt_attention.md,

        layer_idx: int
            The index of this attention layer, used to access kv_cache_block_offsets,

        num_heads: int
            The number of heads,

        num_kv_heads: int
            The number of KV heads, generic to handle MHA/MQA/GQA,

        hidden_size_per_head: int
            The hidden size per head,

        q_scaling: float
            The value used to compute the scaling factor applied to the output
            of the Q*K^T product. See Scaling Factors in docs/gpt_attention.md,

        qk_tanh_scale: float
            The scale * tanh(value / scale) used to compute the scaling factor applied to the output
            of the Q*K^T product. Note this is only used by grok models.

        rotary_embedding_dim: int
            The dimension to compute RoPE. Use 0 when position_embedding_type is not RoPE.

        rotary_embedding_base: float
            The theta value to use for RoPE. Ignored when position_embedding_type is not RoPE.

        rotary_embedding_scale_type: RotaryScalingType
            The scaling type of RoPE. Ignored when position_embedding_type is not RoPE.
            Possible rotary scaling type:
                * RotaryScalingType.none
                * RotaryScalingType.linear
                * RotaryScalingType.dynamic
                * RotaryScalingType.longrope
                * RotaryScalingType.llama3

        rotary_embedding_scale: float
            The scale value to use for linear/dynamic scaling in RoPE.
            Ignored when position_embedding_type is not RoPE.
            Must be set to 1 (default) if rotary_embedding_scale_type is `none`.

        rotary_inv_freq: float Tensor
            The rotary inv freq with shape [head_size / 2].

        rotary_cos_sin: float2(cos/sin) Tensor
            The rotary cos/sin cache, which will be reused among different requests.
            It is taken as constant tensor.

        rotary_embedding_max_positions: int
            Needed only for `dynamic` RoPE scaling. Ignored otherwise.

        position_embedding_type: PositionEmbeddingType
            The position embedding type:
                * PositionEmbeddingType.learned_absolute
                * PositionEmbeddingType.relative
                * PositionEmbeddingType.rope_gptj
                * PositionEmbeddingType.rope_gpt_neox
                * PositionEmbeddingType.alibi
                * PositionEmbeddingType.alibi_with_scale

        kv_orig_quant_scale: Tensor
            The tensor to store the scaling factor for quantization to INT8/FP8
            in the KV cache. Its shape is [1]. See INT8/FP8 KV Cache in
            docs/gpt_attention.md,

        kv_quant_orig_scale: Tensor
            The tensor to store the scaling factor for dequantization from
            INT8/FP8 in the KV cache. Its shape is [1]. See INT8/FP8 KV Cache
            in docs/gpt_attention.md,

        attention_output_orig_quant_scale: Tensor
            The tensor to store the scaling factor for quantization to FP8
            in the KV cache. Its shape is [1].

        kv_cache_quant_mode: QuantMode (int flags)
            Do we enable the INT8 or FP8 KV cache?

        max_context_length: int32_t
            The length of the longest input sequence. See QKV Input in
            docs/gpt_attention.md,

        mask_type: int = 1
            The type of mask:
                * tensorrt_llm.layers.AttentionMaskType.padding for BERT,
                * tensorrt_llm.layers.AttentionMaskType.causal for GPT,
                * tensorrt_llm.layers.AttentionMaskType.sliding_window_causal for GPT,
                * tensorrt_llm.layers.AttentionMaskType.bidirectional for ChatGLM-6B,
                * tensorrt_llm.layers.AttentionMaskType.bidirectionalglm for GLM-10B,
                * tensorrt_llm.layers.AttentionMaskType.blocksparse for Phi-3-small,
                * tensorrt_llm.layers.AttentionMaskType.custom_mask for any models.

        block_sparse_block_size: int
            Block size in block sparse attention

        block_sparse_homo_head_pattern: bool
            Do all attention heads share same vertical stride pattern?

        block_sparse_num_local_blocks: int
            Number of active blocks near diagonal

        block_sparse_vertical_stride: int
            Stride of active blocks in vertical dimension

        alibi_slopes: Tensor
            The ALiBi slopes. The ALiBi bias is computed on-the-fly in the kernel
            when possible,

        tp_size: int
            The number of processes/GPUs when tensor parallelism is activated,

        tp_rank: int
            The rank of that process (when running tensor parallelism),

        kv_cache_block_offsets:
            The tensor of block offsets for the KV cache. Its shape is
            [num_layers, max_batch_size, max_beam_width, 2, max_blocks_per_sequence * 2],
            See KV cache section in docs/gpt_attention.md, on gpu,

        host_kv_cache_block_offsets:
            The same as kv_cache_block_offsets, but on cpu,

        host_kv_cache_pool_pointers:
            The tensor of pool pointers for the KV cache. Its shape is [2],
            See KV cache section in docs/gpt_attention.md, on gpu,

        do_cross_attention: bool = False
            Do we use this as cross attention instead of self attention,

        cross_qkv: Tensor = None
            The QKV tensor of encoder output hidden states. Its shape is [batch_size, max_seqlen, 3
            * hidden_dim] in padded mode and [1, num_tokens, 3 * hidden_dim] in
            packed mode,

        cross_qkv_length: Tensor = None
            The length of the longest encoder output sequence,

        encoder_input_lengths: Tensor
            The tensor that stores the length of each encoder input sequence. Its shape is [batch_size],

        relative_attention_bias: Tensor = None
            The relative attention bias [num_heads, max_seq_len, max_seq_len], or The relative attention embedding table for implicit mode, [num_heads, num_buckets].

        max_distance: int = 0
            The maximum distance of relative position in attention, for implicit mode.
            Default value is 0, meaning to use the regular mode of relative attention bias.
            Implicit mode is only enabled when passing in non-zero positive max_distance value.
            See relative attention bias in docs/gpt_attention.md

        host_context_lengths: Tensor = None (On CPU)
            A host tensor that contains the lengths of the different inputs,

        qkv_bias: Tensor = None,
            The qkv bias tensor.

        use_cache: bool = False
            Do we need to store kv cache ? not needed if there is no generation phase.

        spec_decoding_is_generation_length_variable: bool = False,
            Whether the generation lengths can be different for each sequence in a batch.
            For Medusa, this should be set False.
            For Redrafter, this should be set to True.

        spec_decoding_max_generation_length: int = 1,
            The maximum number of tokens possible in the generation phase per sequence.

        spec_decoding_generation_lengths: Tensor = None,
            The generation phase tokens' lengths for each sequence.
            Shape: [batch_size]

        spec_decoding_position_offsets: Tensor = None,
            The speculative decoding tokens's position offsets (shared by all sequences).
            Shape: [batch_size, num_draft_tokens + 1].

        spec_decoding_packed_mask: Tensor = None,
            The speculative decoding tokens's attention mask (packed into uint32_t bits).
            remove_input_padding is False:
                Shape: [batch_size, num_draft_tokens + 1, divUp(num_draft_tokens + 1, 32)].
            remove_input_padding is True:
                Shape: [sum(spec_decoding_generation_lengths), divUp(num_draft_tokens + 1, 32)].


        host_runtime_perf_knobs: Tensor = None,
            The runtime perf knobs bit mask, controls whether to use certain perf knob in the runtime.

    Returns:
        The tensor produced by that layer.
    """
    assert host_request_types is not None
    assert (alibi_slopes is not None) == (position_embedding_type.is_alibi())
    attn_plg_creator = trt.get_plugin_registry().get_plugin_creator("GPTAttention", "1", TRT_LLM_PLUGIN_NAMESPACE)
    assert attn_plg_creator is not None
    assert host_context_lengths is not None or not default_net().plugin_config.remove_input_padding
    assert isinstance(max_context_length, int)
    assert host_max_attention_window_sizes is not None
    assert host_sink_token_length is not None

    paged_kv_cache_flag = default_net().plugin_config.paged_kv_cache
    if isinstance(qkv, list):
        is_unfuse_qkv_gemm = 1
    else:
        is_unfuse_qkv_gemm = 0
    unfuse_qkv_gemm = trt.PluginField(
        "unfuse_qkv_gemm", np.array(np.int8(is_unfuse_qkv_gemm), dtype=np.int8), trt.PluginFieldType.INT8
    )

    layer_idx = trt.PluginField("layer_idx", np.array(layer_idx, dtype=np.int32), trt.PluginFieldType.INT32)
    nheads = trt.PluginField("num_heads", np.array(num_heads, dtype=np.int32), trt.PluginFieldType.INT32)
    vision_start = trt.PluginField("vision_start", np.array(vision_start, dtype=np.int32), trt.PluginFieldType.INT32)
    vision_length = trt.PluginField("vision_length", np.array(vision_length, dtype=np.int32), trt.PluginFieldType.INT32)
    num_kv_heads = trt.PluginField("num_kv_heads", np.array(num_kv_heads, dtype=np.int32), trt.PluginFieldType.INT32)
    head_size = trt.PluginField("head_size", np.array(hidden_size_per_head, dtype=np.int32), trt.PluginFieldType.INT32)
    unidirectional = trt.PluginField("unidirectional", np.array(1, dtype=np.int32), trt.PluginFieldType.INT32)
    q_scaling = trt.PluginField("q_scaling", np.array(q_scaling, dtype=np.float32), trt.PluginFieldType.FLOAT32)
    qk_tanh_scale = trt.PluginField(
        "qk_tanh_scale", np.array(qk_tanh_scale, dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    rotary_embedding_dim = trt.PluginField(
        "rotary_embedding_dim", np.array(rotary_embedding_dim, dtype=np.int32), trt.PluginFieldType.INT32
    )
    rotary_embedding_base = trt.PluginField(
        "rotary_embedding_base", np.array(rotary_embedding_base, dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    rotary_embedding_scale_type = trt.PluginField(
        "rotary_embedding_scale_type", np.array(rotary_embedding_scale_type, dtype=np.int8), trt.PluginFieldType.INT8
    )
    rotary_embedding_scale = trt.PluginField(
        "rotary_embedding_scale", np.array(rotary_embedding_scale, dtype=np.float32), trt.PluginFieldType.FLOAT32
    )
    rotary_embedding_short_m_scale = trt.PluginField(
        "rotary_embedding_short_m_scale",
        np.array(rotary_embedding_short_m_scale, dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )
    rotary_embedding_long_m_scale = trt.PluginField(
        "rotary_embedding_long_m_scale",
        np.array(rotary_embedding_long_m_scale, dtype=np.float32),
        trt.PluginFieldType.FLOAT32,
    )
    rotary_embedding_max_positions = trt.PluginField(
        "rotary_embedding_max_positions",
        np.array(rotary_embedding_max_positions, dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    rotary_embedding_original_max_positions = trt.PluginField(
        "rotary_embedding_original_max_positions",
        np.array(rotary_embedding_original_max_positions, dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    position_embedding_type = trt.PluginField(
        "position_embedding_type", np.array(int(position_embedding_type), dtype=np.int8), trt.PluginFieldType.INT8
    )
    context_fmha_type = trt.PluginField(
        "context_fmha_type",
        np.array(np.int8(default_net().plugin_config.context_fmha_type), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    remove_input_padding = trt.PluginField(
        "remove_input_padding",
        np.array(np.int8(default_net().plugin_config.remove_input_padding), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    is_spec_decoding_enabled = trt.PluginField(
        "is_spec_decoding_enabled",
        np.array(np.int8(spec_decoding_packed_mask is not None), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    spec_decoding_is_generation_length_variable = trt.PluginField(
        "spec_decoding_is_generation_length_variable",
        np.array(np.int8(spec_decoding_is_generation_length_variable), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    spec_decoding_max_generation_length = trt.PluginField(
        "spec_decoding_max_generation_length",
        np.array(spec_decoding_max_generation_length, dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    p_dtype = default_net().plugin_config.gpt_attention_plugin
    pf_type = trt.PluginField(
        "type_id", np.array([int(str_dtype_to_trt(p_dtype))], np.int32), trt.PluginFieldType.INT32
    )
    # reset mask_type to custom_mask.
    if context_fmha_custom_mask is not None:
        mask_type = AttentionMaskType.custom_mask
    mask_type = trt.PluginField("mask_type", np.array([int(mask_type)], np.int32), trt.PluginFieldType.INT32)
    block_sparse_block_size = trt.PluginField(
        "block_sparse_block_size", np.array([block_sparse_block_size], np.int32), trt.PluginFieldType.INT32
    )
    block_sparse_homo_head_pattern = trt.PluginField(
        "block_sparse_homo_head_pattern",
        np.array(np.int8(block_sparse_homo_head_pattern), np.int8),
        trt.PluginFieldType.INT8,
    )
    block_sparse_num_local_blocks = trt.PluginField(
        "block_sparse_num_local_blocks", np.array([block_sparse_num_local_blocks], np.int32), trt.PluginFieldType.INT32
    )
    block_sparse_vertical_stride = trt.PluginField(
        "block_sparse_vertical_stride", np.array([block_sparse_vertical_stride], np.int32), trt.PluginFieldType.INT32
    )
    enable_xqa = trt.PluginField(
        "enable_xqa", np.array(np.int8(default_net().plugin_config.enable_xqa), dtype=np.int8), trt.PluginFieldType.INT8
    )
    tp_size = trt.PluginField("tp_size", np.array(tp_size, dtype=np.int32), trt.PluginFieldType.INT32)
    tp_rank = trt.PluginField("tp_rank", np.array(tp_rank, dtype=np.int32), trt.PluginFieldType.INT32)
    kv_cache_quant_mode_field = trt.PluginField(
        "kv_cache_quant_mode", np.array(kv_cache_quant_mode, dtype=np.int32), trt.PluginFieldType.INT32
    )
    paged_kv_cache = trt.PluginField(
        "paged_kv_cache", np.array(paged_kv_cache_flag, dtype=np.int32), trt.PluginFieldType.INT32
    )
    tokens_per_block = trt.PluginField(
        "tokens_per_block",
        np.array(default_net().plugin_config.tokens_per_block, dtype=np.int32),
        trt.PluginFieldType.INT32,
    )
    max_context_length = trt.PluginField(
        "max_context_length", np.array(max_context_length, np.int32), trt.PluginFieldType.INT32
    )
    pos_shift_enabled = trt.PluginField(
        "pos_shift_enabled",
        np.array(np.int8(default_net().plugin_config.streamingllm), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    dense_context_fmha = trt.PluginField(
        "dense_context_fmha",
        np.array(np.int8(default_net().plugin_config.streamingllm), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    if qkv_bias is None:
        qkv_bias_enabled = trt.PluginField("qkv_bias_enabled", np.array(0, dtype=np.int8), trt.PluginFieldType.INT8)
    else:
        qkv_bias_enabled = trt.PluginField("qkv_bias_enabled", np.array(1, dtype=np.int8), trt.PluginFieldType.INT8)
    do_cross_attention_field = trt.PluginField(
        "do_cross_attention", np.array(np.int8(do_cross_attention), dtype=np.int8), trt.PluginFieldType.INT8
    )
    max_distance = trt.PluginField("max_distance", np.array(max_distance, dtype=np.int32), trt.PluginFieldType.INT32)
    use_paged_context_fmha_field = trt.PluginField(
        "use_paged_context_fmha",
        np.array(np.int8(default_net().plugin_config.use_paged_context_fmha), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    use_fp8_context_fmha_field = trt.PluginField(
        "use_fp8_context_fmha",
        np.array(np.int8(default_net().plugin_config.use_fp8_context_fmha), dtype=np.int8),
        trt.PluginFieldType.INT8,
    )
    use_cache_pf = trt.PluginField("use_cache", np.array([use_cache], dtype=np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection(
        [
            layer_idx,
            nheads,
            vision_start,
            vision_length,
            num_kv_heads,
            head_size,
            unidirectional,
            q_scaling,
            qk_tanh_scale,
            position_embedding_type,
            rotary_embedding_dim,
            rotary_embedding_base,
            rotary_embedding_scale_type,
            rotary_embedding_scale,
            rotary_embedding_short_m_scale,
            rotary_embedding_long_m_scale,
            rotary_embedding_max_positions,
            rotary_embedding_original_max_positions,
            tp_size,
            tp_rank,
            unfuse_qkv_gemm,
            context_fmha_type,
            enable_xqa,
            kv_cache_quant_mode_field,
            remove_input_padding,
            mask_type,
            block_sparse_block_size,
            block_sparse_homo_head_pattern,
            block_sparse_num_local_blocks,
            block_sparse_vertical_stride,
            paged_kv_cache,
            tokens_per_block,
            pf_type,
            max_context_length,
            qkv_bias_enabled,
            do_cross_attention_field,
            max_distance,
            pos_shift_enabled,
            dense_context_fmha,
            use_paged_context_fmha_field,
            use_fp8_context_fmha_field,
            use_cache_pf,
            is_spec_decoding_enabled,
            spec_decoding_is_generation_length_variable,
            spec_decoding_max_generation_length,
        ]
    )

    attn_plug = attn_plg_creator.create_plugin("causal_attn", pfc)
    assert attn_plug
    plug_inputs = [*qkv] if is_unfuse_qkv_gemm else [qkv]
    if context_fmha_custom_mask is not None:
        plug_inputs += [context_fmha_custom_mask]
    if use_cache:
        plug_inputs += [
            sequence_length,
            host_past_key_value_lengths,
            host_max_attention_window_sizes,
            host_sink_token_length,
            context_lengths,
            cache_indirection,
            host_request_types,
        ]
    else:
        plug_inputs += [
            host_max_attention_window_sizes,
            host_sink_token_length,
            context_lengths,
            host_request_types,
        ]
    if use_cache:
        if paged_kv_cache_flag:
            assert (
                kv_cache_block_offsets is not None
            ), "Paged kv cache is enabled, the kv_cache_block_offsets tensor shall not be None"
            assert (
                host_kv_cache_block_offsets is not None
            ), "Paged kv cache is enabled, the host_kv_cache_block_offsets tensor shall not be None"
            assert (
                host_kv_cache_pool_pointers is not None
            ), "Paged kv cache is enabled, the host_kv_cache_pool_pointers tensor shall not be None"
            plug_inputs += [kv_cache_block_offsets, host_kv_cache_block_offsets, host_kv_cache_pool_pointers]
        else:
            plug_inputs += [past_key_value]

    if use_cache and kv_cache_quant_mode.has_kv_cache_quant():
        plug_inputs += [kv_orig_quant_scale, kv_quant_orig_scale]

    if attention_output_orig_quant_scale is not None:
        assert default_net().plugin_config.use_fp8_context_fmha, "FP8 Context FMHA needs to be enabled"
        plug_inputs += [attention_output_orig_quant_scale]

    if rotary_inv_freq is not None:
        plug_inputs += [rotary_inv_freq]
    if rotary_cos_sin is not None:
        plug_inputs += [rotary_cos_sin]

    if alibi_slopes is not None:
        plug_inputs += [alibi_slopes]

    if relative_attention_bias is not None:
        plug_inputs += [relative_attention_bias]

    if do_cross_attention:
        plug_inputs += [cross_qkv, cross_qkv_length, encoder_input_lengths]

    if default_net().plugin_config.remove_input_padding:
        plug_inputs += [host_context_lengths]

    if qkv_bias is not None:
        plug_inputs += [qkv_bias]

    if spec_decoding_packed_mask is not None:
        # add position_ids as well only if speculative decoding mode
        assert spec_decoding_position_offsets is not None
        assert spec_decoding_generation_lengths is not None
        plug_inputs += [spec_decoding_generation_lengths, spec_decoding_packed_mask, spec_decoding_position_offsets]
    if host_runtime_perf_knobs is not None:
        plug_inputs += [host_runtime_perf_knobs]

    for idx, i in enumerate(plug_inputs):
        assert i is not None, f"Found None input for {idx} th item in plugin inputs {plug_inputs}"

    plug_inputs = [i.trt_tensor for i in plug_inputs]
    # ============================ patch start ============================
    import os

    if layer_idx.data == 0 and (debug_artifacts_dir := os.environ.get("DEBUG_ARTIFACTS_DIR", None)) is not None:
        with open(artifact_path := os.path.join(debug_artifacts_dir, "plugin.txt"), "w") as f:
            print(f"Writing debug artifact at {artifact_path}")
            f.writelines(
                (
                    "plugin field collection:\n",
                    "\n".join(
                        f"{field.name} ({field.type}): {field.data} "
                        f"(dtype={field.data.dtype}, shape={field.data.shape})"
                        for field in pfc
                    ),
                    "\nplugin inputs:\n",
                    "\n".join(f"ITensor(name={t.name}, dtype={t.dtype.name}, shape={t.shape})" for t in plug_inputs),
                )
            )
    # ============================ patch end ============================
    layer = default_trtnet().add_plugin_v2(plug_inputs, attn_plug)
    _add_plugin_info(layer, attn_plg_creator, "causal_attn", pfc)
    output = _create_tensor(layer.get_output(0), layer)
    present_key_value = None
    if use_cache and not paged_kv_cache_flag:
        present_key_value = _create_tensor(layer.get_output(1), layer)
        assert present_key_value is not None
        expected_outputs = 2
    else:
        expected_outputs = 1

    assert (
        layer.num_outputs == expected_outputs
    ), f"Plugin outputs number mismatch with expected, got {layer.num_outputs}, expected {expected_outputs}"

    if kv_cache_quant_mode.has_int8_kv_cache() and not default_net().strongly_typed:
        if not paged_kv_cache_flag:
            # past key value
            layer.get_input(8).set_dynamic_range(-127, 127)
            # present key value
            layer.get_output(1).set_dynamic_range(-127, 127)
        else:
            layer.get_input(0).set_dynamic_range(-127, 127)
            layer.get_input(1).set_dynamic_range(-127, 127)
            layer.get_output(0).set_dynamic_range(-127, 127)
    assert output is not None
    return output, present_key_value


# Note gpt_attention patch must be done manually
# trtllm.functional.gpt_attention = patched_gpt_attention

trtllm.Network.to_dot = patched_trtllm_network_to_dot

trtllm.Builder.build_engine = patched_builder_build_engine

RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin = (
    patched_create_sinusoidal_positions_for_attention_plugin
)

GenerationSession.dump_debug_buffers = patched_dump_debug_buffers


logger.info("ditto patches are applied!")
