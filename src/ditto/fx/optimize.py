from collections.abc import Callable

import torch
from loguru import logger
from torch.fx import GraphModule

from ..arguments import TRTLLMArgumentHint
from ..constants import FX_TRANSFORM_MAXIMUM_ITERATION
from ..literals import PassName
from .passes import (
    AddTRTLLMInputs,
    BindUnmatchedLoraProtos,
    CanonicalizeCopy,
    CastMMToFP32,
    ConstantFolding,
    DecomposeAddMM,
    DeferCast,
    DeferUnsqueeze,
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
    FuseEquivalentNodes,
    FuseGatedMLPProjections,
    FuseQKVProjections,
    FuseReciprocalMul,
    HerdConstantsToTheRight,
    IndexLayers,
    InsertGatherLastTokenIds,
    PopLoraPlugins,
    ReplaceIndexBySlice,
    ReplaceMMByFakeGemmPlugin,
    ReplaceSDPAByFakeGPTAttentionPlugin,
    ReplaceViewByReshape,
    RewriteFloatingPointLiteralsAsNodes,
    RewriteReshapeAsUnsqueeze,
    RewriteSplitAsSlices,
    StashLoraSubgraphs,
    WrapRoPESubgraphs,
    WrapSDPASubgraphs,
)
from .passes.defer_unsqueeze import SwapUnsqueezeWithSymSizeInt
from .passes.infra import GraphOptimizationPass, PassManager


def get_optimization_transform(
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
) -> Callable[[GraphModule], GraphModule]:
    """Optimize the given graph module inplace.

    Args:
        argument_hint (TRTLLMArgumentHint): the type hints for TRTLLM inputs
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
            argument_hint,
            dtype,
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
        *transforms: The transforms to compose

    Returns:
        A function that applies all the given transforms to a graph module
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
    FuseEquivalentNodes,
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
    RewriteFloatingPointLiteralsAsNodes,
    RewriteReshapeAsUnsqueeze,
)


def get_trtllm_conversion_transform(
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    run_matmuls_in_fp32: bool = False,
    run_activations_in_model_dtype: bool = True,
) -> Callable[[GraphModule], GraphModule]:
    """Create a transform that converts a graph module to TensorRT-LLM compatible format.

    Args:
        argument_hint: Type hints for TRTLLM inputs
        dtype: Data type for plugins
        skipped_optimizers: Names of optimization passes to skip
        run_matmuls_in_fp32: Whether to run matrix multiplications in FP32
        run_activations_in_model_dtype: Whether to run activations in model dtype

    Returns:
        A function that applies TRT-LLM conversion passes to a graph module
    """
    passes: list[type[GraphOptimizationPass] | GraphOptimizationPass] = [
        AddTRTLLMInputs(argument_hint=argument_hint),
        SwapUnsqueezeWithSymSizeInt,  # required for `InsertGatherLastTokenIds`
        InsertGatherLastTokenIds,
        StashLoraSubgraphs,
        FuseQKVProjections,
        FuseGatedMLPProjections,
        WrapRoPESubgraphs,
        ReplaceIndexBySlice,
        ReplaceSDPAByFakeGPTAttentionPlugin(dtype=dtype),
        IndexLayers,
        BindUnmatchedLoraProtos,
        PopLoraPlugins(argument_hint=argument_hint),
        ReplaceMMByFakeGemmPlugin,
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
        skipped_optimizers: Names of optimization passes to skip

    Returns:
        A function that applies level 1 optimization passes to a graph module
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
        skipped_optimizers: Names of optimization passes to skip

    Returns:
        A function that applies level 2 optimization passes to a graph module
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
        logger.warning(f"Unrecognized skipped optmizer names: {skipped_optimizers}")

    return pass_manager.as_transform()
