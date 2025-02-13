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

from typing import Literal

AttnQKVPrefix = Literal["attn_q", "attn_k", "attn_v"]

AttnGatedMLPPrefix = Literal["mlp_h_to_4h", "mlp_gate"]

DTypeLiteral = Literal[
    "float16",
    "bfloat16",
    "float32",
    "int8",
    "int32",
    "int64",
    "bool",
    "fp8",
]
"""The available data types are listed in `_str_to_trt_dtype_dict` from `tensorrt_llm._utils`
which is used by the function `str_dtype_to_trt` in the same file."""

KVCacheTypeLiteral = Literal["CONTINUOUS", "DISABLED", "PAGED"]

LoraCheckpointLiteral = Literal["hf", "nemo"]

LoraStateDictSuffix = Literal["lora_A.weight", "lora_B.weight"]

LoraPluginInputPrefix = Literal[
    "attn_qkv",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_dense",
    "mlp_h_to_4h",
    "mlp_4h_to_h",
    "mlp_gate",
    "cross_attn_qkv",
    "cross_attn_q",
    "cross_attn_k",
    "cross_attn_v",
    "cross_attn_dense",
    "moe_h_to_4h",
    "moe_4h_to_h",
    "moe_gate",
    "moe_router",
    "mlp_router",
]

PassName = Literal[
    "AddTRTLLMInputs",
    "BindUnmatchedLoraProtos",
    "CanonicalizeCopy",
    "CastMMToFP32",
    "ConstantFolding",
    "DecomposeAddMM",
    "DeferCast",
    "DeferUnsqueeze",
    "EliminateCommonExpressions",
    "EliminateNopCatOrStack",
    "EliminateNopPermute",
    "EliminateNopReshapeOrExpand",
    "EliminateNopSlice",
    "EliminateUnsqueezeSqueeze",
    "FixActivationPrecision",
    "FixBinaryElementwiseOpOverloads",
    "FixSliceRanges",
    "ForgetSubmodules",
    "FuseConsecutivePermutes",
    "FuseConsecutiveReshapes",
    "FuseConsecutiveSliceConcat",
    "FuseConsecutiveSplitConcat",
    "FuseConsecutiveToCopys",
    "FuseGatedMLPProjections",
    "FuseQKVProjections",
    "FuseReciprocalMul",
    "HerdConstantsToTheRight",
    "IndexLayers",
    "InsertGatherLastTokenIds",
    "ParallelizeLinear",
    "ParallelizePipeline",
    "PopLoraPlugins",
    "PropagateTensorParallelism",
    "ReplaceMMByGemmPlugin",
    "ReplaceSDPAByGPTAttentionPlugin",
    "ReplaceViewByReshape",
    "ResetCodeGen",
    "ResolveDynamicReshape",
    "RewriteFloatingPointLiteralsAsNodes",
    "RewriteReshapeAsUnsqueeze",
    "RewriteSplitAsSlices",
    "StashLoraSubgraphs",
    "WrapRoPESubgraphs",
    "WrapSDPASubgraphs",
]
"""The possible names of FX optimization passes"""

PluginFlag = Literal["auto", "float16", "bfloat16", "fp8", None]

QuantAlgoLiteral = Literal[
    "W8A16",
    "W4A16",
    "W4A16_AWQ",
    "W4A8_AWQ",
    "W4A16_GPTQ",
    "W8A8_SQ_PER_CHANNEL",
    "W8A8_SQ_PER_TENSOR_PLUGIN",
    "W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN",
    "W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN",
    "W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN",
    "FP8",
    "FP8_PER_CHANNEL_PER_TOKEN",
    "INT8",
]
