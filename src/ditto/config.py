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

# The input token tensor in most of the decoder-only models from HF are named as "input_ids".
# If this is not the case for your model, you need to change the value of this constant accordingly.
INPUT_IDS: str = os.environ.get("INPUT_IDS", "input_ids")

# In TRT-LLM, the `input_ids` is a 1-dimensional tensor, whereas in HF, it is 2-dimensional with shape (B, S).
# This constant determines in which dimension should the `input_ids` be expanded.
INPUT_IDS_UNSQUEEZE_DIM: Literal[0, 1] = 1 if os.environ.get("INPUT_IDS_UNSQUEEZE_DIM", "0") == "1" else 0
