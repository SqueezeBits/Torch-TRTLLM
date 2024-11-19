# pylint: disable=no-member
import os
from typing import Literal

import tensorrt as trt
import torch

PassName = Literal[
    "CastFP16MMToFP32",
    "ConstantSharing",
    "DeferUnsqueeze",
    "EliminateCopy",
    "EliminateNopCatOrStack",
    "EliminateNopPermute",
    "EliminateNopReshape",
    "EliminateNopSlice",
    "EliminateUnsqueezeSqueeze",
    "EliminateUnusedWeights",
    "FixSliceRanges",
    "FuseConsecutivePermutes",
    "FuseConsecutiveReshapes",
    "FuseConsecutiveSliceConcat",
    "FuseConsecutiveSplitConcat",
    "FuseConsecutiveToCopys",
    "FuseEquivalentNodes",
    "FuseMMConstSiblings",
    "FuseReciprocalMul",
    "InsertGatherLastTokenIds",
    "MakeWeightsContiguous",
    "ReplaceSDPAByFakeGPTAttentionPlugin",
    "ReplaceSDPAByFakeGPTAttentionPluginV2",
    "ReplaceViewByReshape",
    "RewriteMMAsTransposedMM",
    "RewriteReshapeAsUnsqueeze",
    "WrapRoPESubgraphs",
]
"""The possible names of FX optimization passes"""

AUTO_DETECT_ROPE_SUBGRAPH: bool = os.getenv("AUTO_DETECT_ROPE_SUBGRAPH", "1") == "1"
"""
Whether to automatically detect RoPE subgraph and fuse them in GPTAttentionPlugin.
"""

# pylint: disable-next=invalid-envvar-default
DEBUG_ARTIFACTS_DIR: str | None = os.getenv("DEBUG_ARTIFACTS_DIR", None)
"""The directory to save the debug artifacts such as graph module code."""

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""
The default device for the PyTorch modules and tensors.
"""

DEFAULT_TRT_PROFILING_VERBOSITY: trt.ProfilingVerbosity
"""
The default profiling verbosity for TRT engines.
"""
try:
    DEFAULT_TRT_PROFILING_VERBOSITY = (
        PROFILING_VERBOSITIES := {
            "DETAILED": trt.ProfilingVerbosity.DETAILED,
            "LAYER_NAMES_ONLY": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
            "NONE": trt.ProfilingVerbosity.NONE,
        }
    )[os.getenv("DEFAULT_TRT_PROFILING_VERBOSITY", "LAYER_NAMES_ONLY" if DEBUG_ARTIFACTS_DIR is None else "DETAILED")]
except KeyError as e:
    # pylint: disable-next=used-before-assignment
    raise ValueError(f"DEFAULT_TRT_PROFILING_VERBOSITY must be one of {', '.join(PROFILING_VERBOSITIES.keys())}") from e

FX_TRANSFORM_MAXIMUM_ITERATION = int(os.getenv("FX_TRANSFORM_MAXIMUM_ITERATION", "100"))
"""Maximum iteration limit for FX graph transformations."""

GPT_ATTENTION_PLUGIN_DTYPE: torch.dtype = torch.float16
"""The precision for the GPT attention plugin"""

INPUT_IDS: str = os.getenv("INPUT_IDS", "input_ids")
"""
The input token tensor in most of the decoder-only models from HF are named as "input_ids".
If this is not the case for your model, you need to change the value of this constant accordingly.
"""

INPUT_IDS_UNSQUEEZE_DIM: Literal[0, 1] = 1 if os.getenv("INPUT_IDS_UNSQUEEZE_DIM", "0") == "1" else 0
"""
In TRT-LLM, the `input_ids` is a 1-dimensional tensor, whereas in HF, it is 2-dimensional with shape (B, S).
This constant determines in which dimension should the `input_ids` be expanded.
"""

MATMUL_FUSION_MAX_OUTPUT_SIZE: int = int(os.getenv("MATMUL_FUSION_MAX_OUTPUT_SIZE", "16384"))
"""
If there are fusible matmul siblings with the total output dimention size larger than this number,
they will not be fused by the pass `FuseMMConstSiblings`.
If this value is negative, all matmul siblings will be fused.
"""

if "LOGURU_LEVEL" not in os.environ:
    os.environ["LOGURU_LEVEL"] = os.getenv("DITTO_LOG_LEVEL", "INFO")


# Temporary workaround for config compilation
TRTLLM_LLAMA2_7B_CONFIG = """{
    "version": "0.13.0",
    "pretrained_config": {
        "mlp_bias": false,
        "attn_bias": false,
        "rotary_base": 10000.0,
        "rotary_scaling": null,
        "residual_mlp": false,
        "disable_weight_only_quant_plugin": false,
        "moe": {
            "num_experts": 0,
            "top_k": 0,
            "normalization_mode": null,
            "sparse_mixer_epsilon": 0.01,
            "tp_mode": 0
        },
        "remove_duplicated_kv_heads": false,
        "architecture": "LlamaForCausalLM",
        "dtype": "float16",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "hidden_act": "silu",
        "logits_dtype": "float32",
        "norm_epsilon": 1e-05,
        "position_embedding_type": "rope_gpt_neox",
        "max_position_embeddings": 4096,
        "num_key_value_heads": 32,
        "intermediate_size": 11008,
        "mapping": {
            "world_size": 1,
            "gpus_per_node": 8,
            "cp_size": 1,
            "tp_size": 1,
            "pp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1
        },
        "quantization": {
            "quant_algo": null,
            "kv_cache_quant_algo": null,
            "group_size": 128,
            "smoothquant_val": 0.5,
            "clamp_val": null,
            "has_zero_point": false,
            "pre_quant_scale": false,
            "exclude_modules": null
        },
        "use_parallel_embedding": false,
        "embedding_sharding_dim": 0,
        "share_embedding_table": false,
        "head_size": 128,
        "qk_layernorm": false
    },
    "build_config": {
        "max_input_len": 1024,
        "max_seq_len": 4096,
        "opt_batch_size": null,
        "max_batch_size": 256,
        "max_beam_width": 1,
        "max_num_tokens": 8192,
        "opt_num_tokens": 256,
        "max_prompt_embedding_table_size": 0,
        "kv_cache_type": "PAGED",
        "gather_context_logits": false,
        "gather_generation_logits": false,
        "strongly_typed": true,
        "builder_opt": null,
        "force_num_profiles": null,
        "profiling_verbosity": "layer_names_only",
        "enable_debug_output": false,
        "max_draft_len": 0,
        "speculative_decoding_mode": 1,
        "use_refit": false,
        "input_timing_cache": null,
        "output_timing_cache": "model.cache",
        "lora_config": {
            "lora_dir": [],
            "lora_ckpt_source": "hf",
            "max_lora_rank": 64,
            "lora_target_modules": [],
            "trtllm_modules_to_hf_modules": {}
        },
        "auto_parallel_config": {
            "world_size": 1,
            "gpus_per_node": 8,
            "cluster_key": "NVIDIA-RTX-A6000",
            "cluster_info": {
                "inter_node_bw_per_device": 25,
                "intra_node_bw_per_device": 5,
                "inter_node_latency": 10,
                "intra_node_latency": 10,
                "intra_node_sharp": false,
                "inter_node_sharp": true,
                "memory_bw": 768,
                "memory_budget_per_device": 47,
                "math_throughput": {
                    "int4": 722,
                    "int8": 361,
                    "fp8": 0,
                    "float16": 180,
                    "bfloat16": 180,
                    "float32": 90
                },
                "memory_efficiency": 1.0,
                "math_efficiency": 1.0,
                "communication_efficiency": 1.0
            },
            "sharding_cost_model": "alpha_beta",
            "comm_cost_model": "alpha_beta",
            "enable_pipeline_parallelism": false,
            "enable_shard_unbalanced_shape": false,
            "enable_shard_dynamic_shape": false,
            "enable_reduce_scatter": true,
            "builder_flags": null,
            "debug_mode": false,
            "infer_shape": true,
            "validation_mode": false,
            "same_buffer_io": {
                "past_key_value_(\\\\d+)": "present_key_value_\\\\1"
            },
            "same_spec_io": {},
            "sharded_io_allowlist": [
                "past_key_value_\\\\d+",
                "present_key_value_\\\\d*"
            ],
            "fill_weights": false,
            "parallel_config_cache": null,
            "profile_cache": null,
            "dump_path": null,
            "debug_outputs": []
        },
        "weight_sparsity": false,
        "weight_streaming": false,
        "plugin_config": {
            "dtype": "float16",
            "bert_attention_plugin": "auto",
            "gpt_attention_plugin": "auto",
            "gemm_plugin": null,
            "gemm_swiglu_plugin": null,
            "fp8_rowwise_gemm_plugin": null,
            "smooth_quant_gemm_plugin": null,
            "identity_plugin": null,
            "layernorm_quantization_plugin": null,
            "rmsnorm_quantization_plugin": null,
            "nccl_plugin": null,
            "lookup_plugin": null,
            "lora_plugin": null,
            "weight_only_groupwise_quant_matmul_plugin": null,
            "weight_only_quant_matmul_plugin": null,
            "quantize_per_token_plugin": false,
            "quantize_tensor_plugin": false,
            "moe_plugin": "auto",
            "mamba_conv1d_plugin": "auto",
            "low_latency_gemm_plugin": null,
            "context_fmha": true,
            "bert_context_fmha_fp32_acc": false,
            "paged_kv_cache": true,
            "remove_input_padding": true,
            "reduce_fusion": false,
            "enable_xqa": true,
            "tokens_per_block": 64,
            "use_paged_context_fmha": false,
            "use_fp8_context_fmha": false,
            "multiple_profiles": false,
            "paged_state": false,
            "streamingllm": false,
            "manage_weights": false,
            "use_fused_mlp": true
        },
        "use_strip_plan": false,
        "max_encoder_input_len": 1024,
        "use_fused_mlp": "enable"
    }
}"""
