from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

from ..config import FX_TRANSFORM_MAXIMUM_ITERATION, SKIPPED_OPTIMIZERS, PassName
from .passes import (
    ConstantSharing,
    DeferUnsqueeze,
    EliminateCopy,
    EliminateEmptyTensorsFromCatOrStack,
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
    ReplaceOperatorSubByATenSub,
    ReplaceSDPAByFakeGPTAttentionPlugin,
    RewriteReshapeAsUnsqueeze,
    WrapRoPESubgraphs,
)


def optimize(graph_module: GraphModule) -> GraphModule:
    """Optimize the given graph module inplace.

    Args:
        graph_module (GraphModule): a graph module
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        GraphModule: the graph module itself transformed in-place
    """
    result = get_pass_manager()(graph_module)
    return result.graph_module


def get_pass_manager(skipped_optimizers: list[PassName] | None = None) -> PassManager:
    """Get pass manager.

    Args:
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        PassManager: a pass manager
    """
    pass_manager = PassManager(steps=FX_TRANSFORM_MAXIMUM_ITERATION)

    skipped_optimizers = skipped_optimizers or SKIPPED_OPTIMIZERS
    for fx_pass in (
        # optimizations before TRT-LLM transforms
        ConstantSharing,
        EliminateEmptyTensorsFromCatOrStack,
        EliminateNopCatOrStack,
        EliminateCopy,
        EliminateNopSlice,
        FuseConsecutiveReshapes,
        FuseConsecutivePermutes,
        FuseEquivalentNodes,
        EliminateNopReshape,
        EliminateNopPermute,
        EliminateUnsqueezeSqueeze,
        ReplaceOperatorSubByATenSub,
        # TRT-LLM transforms
        InsertGatherLastTokenIds,
        WrapRoPESubgraphs,
        ReplaceSDPAByFakeGPTAttentionPlugin,
        # optimizations after TRT-LLM transforms
        FuseMMConstSiblings,
        # CastMMConstToFP32,
        FuseConsecutiveSplitConcat,
        DeferUnsqueeze,
        RewriteReshapeAsUnsqueeze,
    ):
        if fx_pass.__name__ in skipped_optimizers:
            continue
        pass_manager.add_pass(fx_pass())

    return pass_manager
