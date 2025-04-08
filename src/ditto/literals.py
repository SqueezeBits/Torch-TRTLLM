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
    "uint8",
    "int32",
    "int64",
    "bool",
    "fp8",
]
"""The available data types are listed in `_str_to_trt_dtype_dict` from `tensorrt_llm._utils`
which is used by the function `str_dtype_to_trt` in the same file."""

KVCacheTypeLiteral = Literal["CONTINUOUS", "DISABLED", "PAGED"]

LinearTypeLiteral = Literal[
    "router",
    "shared_expert",
    "shared_expert_gate",
    "mla_kv_a_proj",
    "mla_kv_b_proj",
    "mla_q_a_proj",
    "mla_q_b_proj",
    "mla_o_proj",
]

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
    "CanonicalizeMoEAllReduces",
    "CastMMToFP32",
    "CastRouterToFP32",
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
    "FuseDequantizes",
    "FuseGatedMLPProjections",
    "FuseMLAQKVProjections",
    "FuseQKVProjections",
    "FuseReciprocalMul",
    "HerdConstantsToTheRight",
    "IndexLayers",
    "InsertGatherLastTokenIds",
    "MarkRouters",
    "OverrideMulScalarTypePromotion",
    "ParallelizeLinear",
    "ParallelizePipeline",
    "PopLoraPlugins",
    "PropagateTensorParallelism",
    "ReplaceMoEByMoEPlugin",
    "ReplaceMMByFp8GemmPlugin",
    "ReplaceMMByFp8RowwiseGemmPlugin",
    "ReplaceMMByGemmPlugin",
    "ReplaceMMByWoQGemmPlugin",
    "ReplaceRmsNormByFp8RmsNormPlugin",
    "ReplaceSDPAByGPTAttentionPlugin",
    "ReplaceTopkByTopkLastDimPlugin",
    "ReplaceViewByReshape",
    "ResetCodeGen",
    "ResolveDynamicReshape",
    "RewriteFloatingPointLiteralsAsNodes",
    "RewriteReshapeAsUnsqueeze",
    "RewriteSplitAsSlices",
    "StashActQuantSubgraphs",
    "StashLoraSubgraphs",
    "WrapWeightDequantSubgraphs",
    "WrapRoPESubgraphs",
    "WrapSDPASubgraphs",
]
"""The possible names of FX optimization passes"""

PluginFlag = Literal["auto", "float16", "bfloat16", "fp8", None]

LogLevelLiteral = Literal[
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
]
