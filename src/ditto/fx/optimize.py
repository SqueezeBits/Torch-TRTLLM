from collections.abc import Callable

from loguru import logger
import torch
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from ..arguments import TRTLLMArgumentHint
from ..constants import AUTO_DETECT_ROPE_SUBGRAPH, FX_TRANSFORM_MAXIMUM_ITERATION, PassName
from .passes import (
    AddTRTLLMInputs,
    CastTypeMMToFP32,
    DeferUnsqueeze,
    EliminateCopy,
    EliminateNopCatOrStack,
    EliminateNopPermute,
    EliminateNopReshape,
    EliminateNopSlice,
    EliminateUnsqueezeSqueeze,
    FixActivationPrecision,
    FixSliceRanges,
    FuseConsecutivePermutes,
    FuseConsecutiveReshapes,
    FuseConsecutiveSliceConcat,
    FuseConsecutiveSplitConcat,
    FuseConsecutiveToCopys,
    FuseEquivalentNodes,
    FuseMMConstSiblings,
    FuseReciprocalMul,
    HerdConstantsToTheRight,
    InsertGatherLastTokenIds,
    ReplaceMMByFakeGemmPlugin,
    ReplaceSDPAByFakeGPTAttentionPlugin,
    ReplaceSDPAByFakeGPTAttentionPluginV2,
    ReplaceViewByReshape,
    RewriteReshapeAsUnsqueeze,
    WrapRoPESubgraphs,
)
from .passes.defer_unsqueeze import SwapUnsqueezeWithSymSizeInt
from .passes.graph_pass import GraphOptimizationPass


def get_optimization_transform(
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    allow_matmul_in_fp16: bool = False,
    allow_activation_in_fp16: bool = True,
) -> Callable[[GraphModule], GraphModule]:
    """Optimize the given graph module inplace.

    Args:
        argument_hint (TRTLLMArgumentHint): the type hints for TRTLLM inputs
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.
        allow_matmul_in_fp16 (bool, optional): whether to allow matrix multiplication to be performed in FP16 precision.
            Defaults to False.
        allow_activation_in_fp16 (bool, optional): whether to allow activations (a.k.a. non-linearities) to be
            performed in FP16 precision. Defaults to True.

    Returns:
        Callable[[GraphModule], GraphModule]: the function that applies FX optimization passes to the given graph module
    """
    return compose(
        get_level1_transform(skipped_optimizers),
        get_trtllm_conversion_transform(
            argument_hint,
            dtype,
            skipped_optimizers=skipped_optimizers,
            allow_matmul_in_fp16=allow_matmul_in_fp16,
            allow_activation_in_fp16=allow_activation_in_fp16,
        ),
        get_level2_transform(skipped_optimizers),
    )


def compose(*transforms: Callable[[GraphModule], GraphModule]) -> Callable[[GraphModule], GraphModule]:
    def composed_transform(graph_module: GraphModule) -> GraphModule:
        for transform in transforms:
            graph_module = transform(graph_module)
        return graph_module

    return composed_transform


# conversions required for TRT-LLM engine
TRTLLM_CONVERSION_PASSES: tuple[type[GraphOptimizationPass], ...] = (
    (
        InsertGatherLastTokenIds,
        WrapRoPESubgraphs,
        ReplaceSDPAByFakeGPTAttentionPlugin,
        FuseMMConstSiblings,
        # ReplaceMMByFakeGemmPlugin,
    )
    if AUTO_DETECT_ROPE_SUBGRAPH
    else (
        InsertGatherLastTokenIds,
        ReplaceSDPAByFakeGPTAttentionPluginV2,
        # TODO: improve memory management of the pass `FuseMMConstSiblings`
        FuseMMConstSiblings,
        # ReplaceMMByFakeGemmPlugin,
    )
)

# passes required before the TRT-LLM conversion passes
LEVEL1_PASSES: tuple[type[GraphOptimizationPass], ...] = (
    # TODO: improve memory management of the pass `ConstantSharing`
    # ConstantSharing,
    EliminateNopCatOrStack,
    EliminateCopy,
    EliminateNopSlice,
    FixSliceRanges,
    FuseConsecutiveReshapes,
    FuseConsecutivePermutes,
    FuseConsecutiveToCopys,
    FuseEquivalentNodes,
    EliminateNopReshape,
    EliminateNopPermute,
    EliminateUnsqueezeSqueeze,
    HerdConstantsToTheRight,
    # TODO: improve memory management of the pass `EliminateUnusedWeights`
    # EliminateUnusedWeights,
    # TODO: improve memory management of the pass `MakeWeightsContiguous`
    # MakeWeightsContiguous,
    ReplaceViewByReshape,
)

# passes required after the TRT-LLM conversion passes
LEVEL2_PASSES: tuple[type[GraphOptimizationPass], ...] = (
    FuseConsecutiveSliceConcat,
    FuseConsecutiveSplitConcat,
    FuseReciprocalMul,
    DeferUnsqueeze,
    RewriteReshapeAsUnsqueeze,
)


def get_trtllm_conversion_transform(
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    allow_matmul_in_fp16: bool = False,
    allow_activation_in_fp16: bool = True,
) -> Callable[[GraphModule], GraphModule]:
    passes: list[type[GraphOptimizationPass] | GraphOptimizationPass] = [
        AddTRTLLMInputs(argument_hint=argument_hint),
        SwapUnsqueezeWithSymSizeInt,  # required for `InsertGatherLastTokenIds`
    ]
    passes.extend(
        (
            InsertGatherLastTokenIds,
            WrapRoPESubgraphs,
            ReplaceSDPAByFakeGPTAttentionPlugin(dtype),
            FuseMMConstSiblings,
            ReplaceMMByFakeGemmPlugin,
        )
        if AUTO_DETECT_ROPE_SUBGRAPH
        else (
            InsertGatherLastTokenIds,
            ReplaceSDPAByFakeGPTAttentionPluginV2(dtype),
            # TODO: improve memory management of the pass `FuseMMConstSiblings`
            FuseMMConstSiblings,
            ReplaceMMByFakeGemmPlugin,
        )
    )
    if not allow_matmul_in_fp16:
        passes.append(CastTypeMMToFP32(dtype))
    if allow_activation_in_fp16:
        passes.append(FixActivationPrecision(dtype=dtype))
    return get_transform(
        *passes,
        skipped_optimizers=skipped_optimizers,
        steps=1,
    )


def get_level1_transform(
    skipped_optimizers: list[PassName] | None = None,
) -> Callable[[GraphModule], GraphModule]:
    return get_transform(
        *LEVEL1_PASSES,
        skipped_optimizers=skipped_optimizers,
    )


def get_level2_transform(
    skipped_optimizers: list[PassName] | None = None,
) -> Callable[[GraphModule], GraphModule]:
    return get_transform(
        *LEVEL1_PASSES,
        *LEVEL2_PASSES,
        skipped_optimizers=skipped_optimizers,
    )


def get_transform(
    *fx_passes: type[GraphOptimizationPass] | GraphOptimizationPass,
    skipped_optimizers: list[PassName] | None = None,
    steps: int = FX_TRANSFORM_MAXIMUM_ITERATION,
) -> Callable[[GraphModule], GraphModule]:
    """Get transform out of the given FX passes.

    Args:
        *fx_passes: (type[GraphOptimizationPass]): the graph optimization pass classes to participate in the transform
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.
        steps (int, optional): the maximum number of iterations until convergence.
            Defaults to FX_TRANSFORM_MAXIMUM_ITERATION.

    Returns:
        PassManager: a pass manager
    """
    pass_manager = PassManager(steps=steps)

    skipped_optimizers = skipped_optimizers or []
    for fx_pass in fx_passes:
        if (
            pass_name := type(fx_pass).__name__ if isinstance(fx_pass, GraphOptimizationPass) else fx_pass.__name__
        ) in skipped_optimizers:
            logger.info(f"Skipping FX optimization pass {pass_name}")
            _ = skipped_optimizers.pop(skipped_optimizers.index(pass_name))  # type: ignore[arg-type]
            continue
        pass_manager.add_pass(fx_pass if isinstance(fx_pass, GraphOptimizationPass) else fx_pass())

    if skipped_optimizers:
        logger.warning(f"Unrecognized skipped optmizer names: {skipped_optimizers}")

    def optimize(graph_module: GraphModule) -> GraphModule:
        result = pass_manager(graph_module)
        if result.modified:
            clean_up_graph_after_modifications(graph_module)
        return result.graph_module

    return optimize
