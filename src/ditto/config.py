import os
from typing import Literal

PassName = Literal[
    "ConstantSharing",
    "EliminateEmptyTensorsFromCatOrStack",
    "EliminateNopCatOrStack",
    "EliminateNopPermute",
    "EliminateNopReshape",
    "EliminateUniqueClone",
    "FuseConsecutivePermutes",
    "FuseConsecutiveReshapes",
    "FuseConsecutiveSplitConcat",
    "FuseEquivalentNodes",
    "FuseMMSiblings",
    "InsertGatherLastTokenIds",
    "ReplaceOperatorSubByATenSub",
    "ReplaceSDPAByFakeGPTAttentionPlugin",
    "WrapRoPESubgraphs",
]

# Maximum iteration limit for FX graph transformations.
FX_TRANSFORM_MAXIMUM_ITERATION = int(os.environ.get("FX_TRANSFORM_MAXIMUM_ITERATION", 100))

MAX_FUSIBLE_MATMUL_OUT_SIZE = int(os.environ.get("MAX_FUSIBLE_MATMUL_OUT_SIZE", 1 << 14))

SKIPPED_OPTIMIZERS: list[PassName] = []
