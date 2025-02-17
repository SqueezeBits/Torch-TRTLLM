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

from collections.abc import Callable

import torch
from loguru import logger
from torch.fx import GraphModule

from ..arguments import TRTLLMArgumentHint
from ..configs import TRTLLMModelConfig
from ..constants import FX_TRANSFORM_MAXIMUM_ITERATION
from ..literals import PassName
from .passes import (
    AddTRTLLMInputs,
    BindUnmatchedLoraProtos,
    CanonicalizeCopy,
    CastMMToFP32,
    CastOutputLogits,
    ConstantFolding,
    DecomposeAddMM,
    DeferCast,
    DeferUnsqueeze,
    EliminateCommonExpressions,
    EliminateNopCatOrStack,
    EliminateNopPermute,
    EliminateNopReshapeOrExpand,
    EliminateNopSlice,
    EliminateUnsqueezeSqueeze,
    FixActivationPrecision,
    FixBinaryElementwiseOpOverloads,
    FixSliceRanges,
    FuseConsecutivePermutes,
    FuseConsecutiveReshapes,
    FuseConsecutiveSliceConcat,
    FuseConsecutiveSplitConcat,
    FuseConsecutiveToCopys,
    FuseGatedMLPProjections,
    FuseQKVProjections,
    FuseReciprocalMul,
    HerdConstantsToTheRight,
    IndexLayers,
    InsertGatherLastTokenIds,
    PopLoraPlugins,
    ReplaceMMByGemmPlugin,
    ReplaceSDPAByGPTAttentionPlugin,
    ReplaceViewByReshape,
    RewriteFloatingPointLiteralsAsNodes,
    RewriteIndexAsSingleSlice,
    RewritePowAsMul,
    RewriteReshapeAsUnsqueeze,
    RewriteSplitAsSlices,
    StashLoraSubgraphs,
    WrapRoPESubgraphs,
    WrapSDPASubgraphs,
)
from .passes.defer_unsqueeze import SwapUnsqueezeWithSymSizeInt
from .passes.infra import GraphOptimizationPass, PassManager


# pylint: disable=duplicate-code
def get_optimization_transform(
    argument_hint: TRTLLMArgumentHint,
    model_config: TRTLLMModelConfig,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
) -> Callable[[GraphModule], GraphModule]:
    """Optimize the given graph module inplace.

    Args:
        argument_hint (TRTLLMArgumentHint): the type hints for TRTLLM inputs
        model_config (TRTLLMModelConfig): Model configurations
        dtype (torch.dtype): the data type for the plugins
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.
        run_matmuls_in_fp32 (bool, optional): whether to run all matrix multiplications in FP32.
            Defaults to False.
        run_activations_in_model_dtype (bool, optional): whether to run all activations (a.k.a. non-linearities) in
            the given `dtype`. Defaults to True.

    Returns:
        Callable[[GraphModule], GraphModule]: the function that applies FX optimization passes to the given graph module
    """
    return compose(
        get_trtllm_conversion_transform(
            argument_hint=argument_hint,
            model_config=model_config,
            dtype=dtype,
            skipped_optimizers=skipped_optimizers,
            run_matmuls_in_fp32=run_matmuls_in_fp32,
            run_activations_in_model_dtype=run_activations_in_model_dtype,
        ),
        get_level2_transform(skipped_optimizers),
        ConstantFolding().as_transform(),
    )


def compose(*transforms: Callable[[GraphModule], GraphModule]) -> Callable[[GraphModule], GraphModule]:
    """Compose multiple transforms into a single transform.

    Args:
        *transforms (Callable[[GraphModule], GraphModule]): The transforms to compose

    Returns:
        Callable[[GraphModule], GraphModule]: A function that applies all the given transforms to a graph module
    """

    def composed_transform(graph_module: GraphModule) -> GraphModule:
        for transform in transforms:
            graph_module = transform(graph_module)
        return graph_module

    return composed_transform


# passes required before the TRT-LLM conversion passes
LEVEL1_PASSES: tuple[type[GraphOptimizationPass], ...] = (
    EliminateNopCatOrStack,
    CanonicalizeCopy,
    EliminateNopSlice,
    FixBinaryElementwiseOpOverloads,
    FixSliceRanges,
    FuseConsecutiveReshapes,
    FuseConsecutivePermutes,
    FuseConsecutiveToCopys,
    EliminateCommonExpressions,
    EliminateNopReshapeOrExpand,
    EliminateNopPermute,
    EliminateUnsqueezeSqueeze,
    HerdConstantsToTheRight,
    ReplaceViewByReshape,
    DecomposeAddMM,
    WrapSDPASubgraphs,
    DeferCast,
    RewriteSplitAsSlices,
)

# passes required after the TRT-LLM conversion passes
LEVEL2_PASSES: tuple[type[GraphOptimizationPass], ...] = (
    FuseConsecutiveSliceConcat,
    FuseConsecutiveSplitConcat,
    FuseReciprocalMul,
    DeferUnsqueeze,
    RewritePowAsMul,
    RewriteFloatingPointLiteralsAsNodes,
    RewriteReshapeAsUnsqueeze,
)


def get_trtllm_output_adaptation_passes(gather_context_logits: bool) -> list[type[GraphOptimizationPass]]:
    """Get the list of graph optimization passes for adapting TensorRT-LLM outputs.

    Args:
        gather_context_logits (bool): Whether to gather all the context logits

    Returns:
        list[type[GraphOptimizationPass]]: A list of graph optimization pass to apply.
    """
    if gather_context_logits:
        return []
    return [
        SwapUnsqueezeWithSymSizeInt,  # required for `InsertGatherLastTokenIds`
        InsertGatherLastTokenIds,
    ]


def get_trtllm_conversion_transform(
    argument_hint: TRTLLMArgumentHint,
    model_config: TRTLLMModelConfig,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
) -> Callable[[GraphModule], GraphModule]:
    """Create a transform that converts a graph module to TensorRT-LLM compatible format.

    Args:
        argument_hint (TRTLLMArgumentHint): Type hints for TRTLLM inputs
        model_config (TRTLLMModelConfig): Model configurations
        dtype (torch.dtype): Data type for plugins
        skipped_optimizers (list[PassName] | None, optional): Names of optimization passes to skip. Defaults to None.
        run_matmuls_in_fp32 (bool, optional): Whether to run matrix multiplications in FP32. Defaults to False.
        run_activations_in_model_dtype (bool, optional): Whether to run activations in model dtype. Defaults to True.

    Returns:
        Callable[[GraphModule], GraphModule]: A function that applies TRT-LLM conversion passes to a graph module
    """
    passes: list[type[GraphOptimizationPass] | GraphOptimizationPass] = [
        AddTRTLLMInputs(argument_hint=argument_hint),
        *get_trtllm_output_adaptation_passes(model_config.gather_context_logits),
        StashLoraSubgraphs,
        FuseQKVProjections,
        FuseGatedMLPProjections,
        WrapRoPESubgraphs,
        RewriteIndexAsSingleSlice,
        ReplaceSDPAByGPTAttentionPlugin(dtype=dtype),
        IndexLayers,
        BindUnmatchedLoraProtos,
        PopLoraPlugins(argument_hint=argument_hint),
        ReplaceMMByGemmPlugin,
        CastOutputLogits(logits_dtype=model_config.logits_dtype),
    ]

    if run_matmuls_in_fp32:
        passes.append(CastMMToFP32)

    if run_activations_in_model_dtype:
        passes.append(FixActivationPrecision(dtype=dtype))

    return get_transform(
        *passes,
        skipped_optimizers=skipped_optimizers,
        steps=1,
        warn_on_partial_convergence=False,
    )


def get_level1_transform(
    skipped_optimizers: list[PassName] | None = None,
) -> Callable[[GraphModule], GraphModule]:
    """Create a transform that applies level 1 optimization passes.

    Args:
        skipped_optimizers (list[PassName] | None, optional): Names of optimization passes to skip. Defaults to None.

    Returns:
        Callable[[GraphModule], GraphModule]: A function that applies level 1 optimization passes to a graph module
    """
    return get_transform(
        *LEVEL1_PASSES,
        skipped_optimizers=skipped_optimizers,
    )


def get_level2_transform(
    skipped_optimizers: list[PassName] | None = None,
) -> Callable[[GraphModule], GraphModule]:
    """Create a transform that applies level 2 optimization passes.

    Args:
        skipped_optimizers (list[PassName] | None, optional): Names of optimization passes to skip. Defaults to None.

    Returns:
        Callable[[GraphModule], GraphModule]: A function that applies level 2 optimization passes to a graph module
    """
    return get_transform(
        *LEVEL1_PASSES,
        *LEVEL2_PASSES,
        skipped_optimizers=skipped_optimizers,
    )


def get_transform(
    *fx_passes: type[GraphOptimizationPass] | GraphOptimizationPass,
    skipped_optimizers: list[PassName] | None = None,
    steps: int = FX_TRANSFORM_MAXIMUM_ITERATION,
    warn_on_partial_convergence: bool = True,
) -> Callable[[GraphModule], GraphModule]:
    """Get transform out of the given FX passes.

    Args:
        *fx_passes: (type[GraphOptimizationPass]): the graph optimization pass classes to participate in the transform
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.
        steps (int, optional): the maximum number of iterations until convergence.
            Defaults to FX_TRANSFORM_MAXIMUM_ITERATION.
        warn_on_partial_convergence (bool, optional): Whether to warn when the graph module doesn't converge
            within the specified `steps`. Defaults to True.

    Returns:
        PassManager: a pass manager
    """
    pass_manager = PassManager(steps=steps, warn_on_partial_convergence=warn_on_partial_convergence)

    skipped_optimizers = skipped_optimizers or []
    for fx_pass in fx_passes:
        if (
            pass_name := type(fx_pass).__name__ if isinstance(fx_pass, GraphOptimizationPass) else fx_pass.__name__
        ) in skipped_optimizers:
            logger.info(f"Skipping FX optimization pass {pass_name}")
            _ = skipped_optimizers.pop(skipped_optimizers.index(pass_name))
            continue
        pass_manager.add_pass(fx_pass)

    if skipped_optimizers:
        logger.warning(f"Unrecognized skipped optimizer names: {skipped_optimizers}")

    return pass_manager.as_transform()
