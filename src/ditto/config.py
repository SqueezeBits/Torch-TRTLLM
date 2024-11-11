import os
from typing import Literal

import torch

PassName = Literal[
    "CastFP16MMToFP32",
    "ConstantSharing",
    "DeferUnsqueeze",
    "EliminateCopy",
    "EliminateNopCatOrStack",
    "EliminateNopPermute",
    "EliminateNopReshape",
    "EliminateNopSlice",
    "EliminateUnsqueezeSqueeze",
    "EliminateUnusedWeights",
    "FuseConsecutivePermutes",
    "FuseConsecutiveReshapes",
    "FuseConsecutiveSliceConcat",
    "FuseConsecutiveSplitConcat",
    "FuseConsecutiveToCopys",
    "FuseEquivalentNodes",
    "FuseMMConstSiblings",
    "FuseReciprocalMul",
    "InsertGatherLastTokenIds",
    "MakeWeightsContiguous",
    "ReplaceSDPAByFakeGPTAttentionPlugin",
    "RewriteMMAsTransposedMM",
    "RewriteReshapeAsUnsqueeze",
    "WrapRoPESubgraphs",
]
"""The possible names of FX optimization passes"""

FX_TRANSFORM_MAXIMUM_ITERATION = int(os.environ.get("FX_TRANSFORM_MAXIMUM_ITERATION", 100))
"""Maximum iteration limit for FX graph transformations."""

INPUT_IDS: str = os.environ.get("INPUT_IDS", "input_ids")
"""
The input token tensor in most of the decoder-only models from HF are named as "input_ids".
If this is not the case for your model, you need to change the value of this constant accordingly.
"""

# In TRT-LLM, the `input_ids` is a 1-dimensional tensor, whereas in HF, it is 2-dimensional with shape (B, S).
# This constant determines in which dimension should the `input_ids` be expanded.
INPUT_IDS_UNSQUEEZE_DIM: Literal[0, 1] = 1 if os.environ.get("INPUT_IDS_UNSQUEEZE_DIM", "0") == "1" else 0
"""
In TRT-LLM, the `input_ids` is a 1-dimensional tensor, whereas in HF, it is 2-dimensional with shape (B, S).
This constant determines in which dimension should the `input_ids` be expanded.
"""


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""
The default device for the PyTorch modules and tensors.
"""

GPT_ATTENTION_PLUGIN_DTYPE: torch.dtype = torch.float16
"""The precision for the GPT attention plugin"""
