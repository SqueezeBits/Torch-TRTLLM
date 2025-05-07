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

from .add_trtllm_inputs import AddTRTLLMInputs
from .bind_unmatched_lora_protos import BindUnmatchedLoraProtos
from .canonicalize_copy import CanonicalizeCopy
from .canonicalize_moe_allreduces import CanonicalizeMoEAllReduces
from .cast_mm_to_fp32 import CastMMToFP32
from .cast_output_logits import CastOutputLogits
from .cast_router_to_fp32 import CastRouterToFP32
from .constant_folding import ConstantFolding
from .decompose_addmm import DecomposeAddMM
from .defer_cast import DeferCast
from .defer_unsqueeze import DeferUnsqueeze
from .eliminate_common_expressions import EliminateCommonExpressions
from .eliminate_nop_cat_or_stack import EliminateNopCatOrStack
from .eliminate_nop_permute import EliminateNopPermute
from .eliminate_nop_reshape_or_expand import EliminateNopReshapeOrExpand
from .eliminate_nop_slice import EliminateNopSlice
from .eliminate_unsqueeze_squeeze import EliminateUnsqueezeSqueeze
from .fix_activation_precision import FixActivationPrecision
from .fix_binary_elementwise_op_overloads import FixBinaryElementwiseOpOverloads
from .fix_slice_ranges import FixSliceRanges
from .forget_submodules import ForgetSubmodules
from .fuse_consecutive_permutes import FuseConsecutivePermutes
from .fuse_consecutive_reshapes import FuseConsecutiveReshapes
from .fuse_consecutive_slice_concat import FuseConsecutiveSliceConcat
from .fuse_consecutive_split_concat import FuseConsecutiveSplitConcat
from .fuse_consecutive_to_copys import FuseConsecutiveToCopys
from .fuse_fake_quantizes import FuseFakeQuantizes
from .fuse_gated_mlp_projections import FuseGatedMLPProjections
from .fuse_qkv_projections import FuseMLAQKVProjections, FuseQKVProjections
from .fuse_reciprocal_mul import FuseReciprocalMul
from .herd_constants_to_the_right import HerdConstantsToTheRight
from .index_layers import IndexLayers
from .insert_gather_last_token_ids import InsertGatherLastTokenIds
from .mark_mla_linears import MarkMLALinears
from .mark_moe_linears import MarkMoELinears
from .override_mul_scalar_type_promotion import OverrideMulScalarTypePromotion
from .parallelize_linear import ParallelizeLinear
from .parallelize_pipeline import ParallelizePipeline
from .pop_lora_plugins import PopLoraPlugins
from .propagate_tensor_parallelism import PropagateTensorParallelism
from .replace_mm_by_fp8_gemm_plugin import ReplaceMMByFp8GemmPlugin
from .replace_mm_by_fp8_rowwise_gemm_plugin import ReplaceMMByFp8RowwiseGemmPlugin
from .replace_mm_by_gemm_plugin import ReplaceMMByGemmPlugin
from .replace_mm_by_woq_gemm_plugin import ReplaceMMByWoQGemmPlugin
from .replace_moe_by_mixture_of_experts_plugin import ReplaceMoEByMoEPlugin
from .replace_rmsnorm_by_fp8_rmsnorm_plugin import ReplaceRmsNormByFp8RmsNormPlugin
from .replace_sdpa_by_gpt_attention_plugin import ReplaceSDPAByGPTAttentionPlugin
from .replace_topk_by_topk_plugin import ReplaceTopkByTopkLastDimPlugin
from .replace_view_by_reshape import ReplaceViewByReshape
from .reset_code_gen import ResetCodeGen
from .resolve_dynamic_reshape import ResolveDynamicReshape
from .rewrite_fp_literals_as_nodes import RewriteFloatingPointLiteralsAsNodes
from .rewrite_index_as_single_slice import RewriteIndexAsSingleSlice
from .rewrite_pow_as_mul import RewritePowAsMul
from .rewrite_reshape_as_unsqueeze import RewriteReshapeAsUnsqueeze
from .rewrite_split_as_slices import RewriteSplitAsSlices
from .stash_activation_fake_quantize import StashActivationFakeQuantize
from .stash_lora_subgraphs import StashLoraSubgraphs
from .stash_output_fake_quantize import StashOutputFakeQuantize
from .wrap_rope_subgraphs import WrapRoPESubgraphs
from .wrap_sdpa_subgraphs import WrapSDPASubgraphs
