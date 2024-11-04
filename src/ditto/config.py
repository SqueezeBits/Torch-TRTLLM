import os
from typing import Literal

PassName = Literal[
    "CastMMConstToFP32",
    "ConstantSharing",
    "DeferUnsqueeze",
    "EliminateCopy",
    "EliminateNopCatOrStack",
    "EliminateNopPermute",
    "EliminateNopReshape",
    "EliminateNopSlice",
    "EliminateUnsqueezeSqueeze",
    "FuseConsecutivePermutes",
    "FuseConsecutiveReshapes",
    "FuseConsecutiveSplitConcat",
    "FuseEquivalentNodes",
    "FuseMMConstSiblings",
    "InsertGatherLastTokenIds",
    "ReplaceSDPAByFakeGPTAttentionPlugin",
    "RewriteReshapeAsUnsqueeze",
    "WrapRoPESubgraphs",
]

# Maximum iteration limit for FX graph transformations.
FX_TRANSFORM_MAXIMUM_ITERATION = int(os.environ.get("FX_TRANSFORM_MAXIMUM_ITERATION", 100))
