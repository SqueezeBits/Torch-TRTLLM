from collections.abc import Callable

from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

from ..config import FX_TRANSFORM_MAXIMUM_ITERATION, PassName
from .passes import (
    CastMMConstToFP32,
    ConstantSharing,
    DeferUnsqueeze,
    EliminateCopy,
    EliminateNopCatOrStack,
    EliminateNopPermute,
    EliminateNopReshape,
    EliminateNopSlice,
    EliminateUnsqueezeSqueeze,
    FuseConsecutivePermutes,
    FuseConsecutiveReshapes,
    FuseConsecutiveSplitConcat,
    FuseEquivalentNodes,
    FuseMMConstSiblings,
    InsertGatherLastTokenIds,
    ReplaceSDPAByFakeGPTAttentionPlugin,
    RewriteReshapeAsUnsqueeze,
    WrapRoPESubgraphs,
)
from .passes.graph_pass import GraphOptimizationPass


def get_optimizer_pass(
    skipped_optimizers: list[PassName] | None = None,
    enforce_projections_in_fp32: bool = False,
) -> Callable[[GraphModule], GraphModule]:
    """Optimize the given graph module inplace.

    Args:
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.
        enforce_projections_in_fp32 (bool, optional): whether to enforce nn.Linear layers' computation in FP32, while
            storing weights in FP16 precision. Defaults to False.

    Returns:
        Callable[[GraphModule], GraphModule]: the function that applies FX optimization passes to the given graph module
    """
    pass_manager = get_pass_manager(skipped_optimizers, enforce_projections_in_fp32)

    def optimize(graph_module: GraphModule) -> GraphModule:
        result = pass_manager(graph_module)
        return result.graph_module

    return optimize


def get_pass_manager(
    skipped_optimizers: list[PassName] | None = None,
    enforce_projections_in_fp32: bool = False,
) -> PassManager:
    """Get pass manager.

    Args:
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.
        enforce_projections_in_fp32 (bool, optional): whether to enforce nn.Linear layers' computation in FP32, while
            storing weights in FP16 precision. Defaults to False.

    Returns:
        PassManager: a pass manager
    """
    pass_manager = PassManager(steps=FX_TRANSFORM_MAXIMUM_ITERATION)

    skipped_optimizers = skipped_optimizers or []
    fx_passes: list[type[GraphOptimizationPass]] = [
        # optimizations before TRT-LLM transforms
        ConstantSharing,
        EliminateNopCatOrStack,
        EliminateCopy,
        EliminateNopSlice,
        FuseConsecutiveReshapes,
        FuseConsecutivePermutes,
        FuseEquivalentNodes,
        EliminateNopReshape,
        EliminateNopPermute,
        EliminateUnsqueezeSqueeze,
        # TRT-LLM transforms
        InsertGatherLastTokenIds,
        WrapRoPESubgraphs,
        ReplaceSDPAByFakeGPTAttentionPlugin,
        # optimizations after TRT-LLM transforms
        FuseMMConstSiblings,
        FuseConsecutiveSplitConcat,
        DeferUnsqueeze,
        RewriteReshapeAsUnsqueeze,
    ]
    if enforce_projections_in_fp32:
        fx_passes.append(CastMMConstToFP32)
    for fx_pass in fx_passes:
        if (pass_name := fx_pass.__name__) in skipped_optimizers:
            print(f"Skipping FX optimization pass {pass_name}")
            continue
        pass_manager.add_pass(fx_pass())

    return pass_manager
