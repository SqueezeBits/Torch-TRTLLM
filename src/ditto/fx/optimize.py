from collections.abc import Callable

from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from ..config import FX_TRANSFORM_MAXIMUM_ITERATION, PassName
from .passes import (
    CastFP16MMToFP32,
    ConstantSharing,
    DeferUnsqueeze,
    EliminateCopy,
    EliminateNopCatOrStack,
    EliminateNopPermute,
    EliminateNopReshape,
    EliminateNopSlice,
    EliminateUnsqueezeSqueeze,
    EliminateUnusedWeights,
    FuseConsecutivePermutes,
    FuseConsecutiveReshapes,
    FuseConsecutiveSliceConcat,
    FuseConsecutiveSplitConcat,
    FuseConsecutiveToCopys,
    FuseEquivalentNodes,
    FuseMMConstSiblings,
    FuseReciprocalMul,
    InsertGatherLastTokenIds,
    MakeWeightsContiguous,
    ReplaceSDPAByFakeGPTAttentionPlugin,
    RewriteMMAsTransposedMM,
    RewriteReshapeAsUnsqueeze,
    WrapRoPESubgraphs,
)
from .passes.graph_pass import GraphOptimizationPass


def get_optimization_transform(
    skipped_optimizers: list[PassName] | None = None,
    *,
    enforce_projections_transposed: bool = False,
    enforce_projections_in_fp32: bool = False,
) -> Callable[[GraphModule], GraphModule]:
    """Optimize the given graph module inplace.

    Args:
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.
        enforce_projections_transposed (bool, optional): whether to enforce nn.Linear layers' computation as transposed
            matmul, storing weights as transposed. Defaults to False.
        enforce_projections_in_fp32 (bool, optional): whether to enforce nn.Linear layers' computation in FP32, while
            storing weights in FP16 precision. Defaults to False.

    Returns:
        Callable[[GraphModule], GraphModule]: the function that applies FX optimization passes to the given graph module
    """
    return compose(
        get_level1_transform(skipped_optimizers),
        get_trtllm_conversion_transform(
            skipped_optimizers,
            enforce_projections_transposed=enforce_projections_transposed,
            enforce_projections_in_fp32=enforce_projections_in_fp32,
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
    InsertGatherLastTokenIds,
    WrapRoPESubgraphs,
    ReplaceSDPAByFakeGPTAttentionPlugin,
    FuseMMConstSiblings,
)

# passes required before the TRT-LLM conversion passes
LEVEL1_PASSES: tuple[type[GraphOptimizationPass], ...] = (
    ConstantSharing,
    EliminateNopCatOrStack,
    EliminateCopy,
    EliminateNopSlice,
    FuseConsecutiveReshapes,
    FuseConsecutivePermutes,
    FuseConsecutiveToCopys,
    FuseEquivalentNodes,
    EliminateNopReshape,
    EliminateNopPermute,
    EliminateUnsqueezeSqueeze,
    EliminateUnusedWeights,
    MakeWeightsContiguous,
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
    skipped_optimizers: list[PassName] | None = None,
    *,
    enforce_projections_transposed: bool = False,
    enforce_projections_in_fp32: bool = False,
) -> Callable[[GraphModule], GraphModule]:
    passes = list(TRTLLM_CONVERSION_PASSES)
    if enforce_projections_transposed:
        passes.append(RewriteMMAsTransposedMM)
    if enforce_projections_in_fp32:
        passes.append(CastFP16MMToFP32)
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
    *fx_passes: type[GraphOptimizationPass],
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
        if (pass_name := fx_pass.__name__) in skipped_optimizers:
            print(f"Skipping FX optimization pass {pass_name}")
            continue
        pass_manager.add_pass(fx_pass())

    def optimize(graph_module: GraphModule) -> GraphModule:
        result = pass_manager(graph_module)
        if result.modified:
            clean_up_graph_after_modifications(graph_module)
        return result.graph_module

    return optimize
