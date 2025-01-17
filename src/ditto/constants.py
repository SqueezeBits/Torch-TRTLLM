import os
from typing import Literal

import tensorrt as trt
import torch
from loguru import logger

PassName = Literal[
    "AddTRTLLMInputs",
    "CanonicalizeCopy",
    "CastMMToFP32",
    "ConstantFolding",
    "DecomposeAddMM",
    "DeferUnsqueeze",
    "EliminateNopCatOrStack",
    "EliminateNopPermute",
    "EliminateNopReshapeOrExpand",
    "EliminateNopSlice",
    "EliminateUnsqueezeSqueeze",
    "FixActivationPrecision",
    "FixBinaryElementwiseOpOverloads",
    "FixSliceRanges",
    "ForgetSubmodules",
    "FuseConsecutivePermutes",
    "FuseConsecutiveReshapes",
    "FuseConsecutiveSliceConcat",
    "FuseConsecutiveSplitConcat",
    "FuseConsecutiveToCopys",
    "FuseEquivalentNodes",
    "FuseQKVProjections",
    "FuseReciprocalMul",
    "InsertGatherLastTokenIds",
    "ParallelizeTensor",
    "HerdConstantsToTheRight",
    "ReplaceMMByFakeGemmPlugin",
    "ReplaceSDPAByFakeGPTAttentionPlugin",
    "ReplaceViewByReshape",
    "ResetCodeGen",
    "RewriteFloatingPointLiteralsAsNodes",
    "RewriteReshapeAsUnsqueeze",
    "WrapRoPESubgraphs",
    "WrapSDPASubgraphs",
]
"""The possible names of FX optimization passes"""

# pylint: disable-next=invalid-envvar-default
DEBUG_ARTIFACTS_DIR: str | None = os.getenv("DEBUG_ARTIFACTS_DIR", None)
"""The directory to save the debug artifacts such as graph module code."""

DEBUG_TENSOR_CHUNK_SIZE: int = int(os.getenv("DEBUG_TENSOR_CHUNK_SIZE", "10"))
"""The number of first/middle/last element values to save for each constant tensor when creating debug artifacts."""

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""
The default device for the PyTorch modules and tensors.
"""

DEFAULT_ONNX_PROTO_SIZE_THRESHOLD: int = int(os.getenv("DEFAULT_ONNX_PROTO_SIZE_THRESHOLD", "0"))
"""
The default size threshold (bytes) for write weights in ONNX as an external data.
"""

DEFAULT_TRT_PROFILING_VERBOSITY: trt.ProfilingVerbosity
"""
The default profiling verbosity for TRT engines.
"""
try:
    if (_verbosity := os.getenv("DEFAULT_TRT_PROFILING_VERBOSITY")) is not None:
        DEFAULT_TRT_PROFILING_VERBOSITY = trt.ProfilingVerbosity.__members__[_verbosity]
    else:
        DEFAULT_TRT_PROFILING_VERBOSITY = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        if DEBUG_ARTIFACTS_DIR is not None:
            DEFAULT_TRT_PROFILING_VERBOSITY = trt.ProfilingVerbosity.DETAILED
            logger.info(
                "Automatically setting 'DEFAULT_TRT_PROFILING_VERBOSITY' to 'DETAILED' as 'DEBUG_ARTIFACT_DIR' is set."
            )
except KeyError as e:
    raise ValueError(
        f"DEFAULT_TRT_PROFILING_VERBOSITY must be one of {', '.join(trt.ProfilingVerbosity.__members__.keys())}"
    ) from e

DISABLE_TRANSFORMER_PATCHES: bool = os.getenv("DISABLE_TRANSFORMER_PATCHES", "0") == "1"
"""Whether to disable the patches for the transformers package."""

FX_TRANSFORM_MAXIMUM_ITERATION = int(os.getenv("FX_TRANSFORM_MAXIMUM_ITERATION", "100"))
"""Maximum iteration limit for FX graph transformations."""

INPUT_IDS: str = os.getenv("INPUT_IDS", "input_ids")
"""
The input token tensor in most of the decoder-only models from HF are named as "input_ids".
If this is not the case for your model, you need to change the value of this constant accordingly.
"""

INPUT_IDS_UNSQUEEZE_DIM: Literal[0, 1] = 1 if os.getenv("INPUT_IDS_UNSQUEEZE_DIM", "0") == "1" else 0
"""
In TRT-LLM, the `input_ids` is a 1-dimensional tensor, whereas in HF, it is 2-dimensional with shape (B, S).
This constant determines in which dimension should the `input_ids` be expanded.
"""

MATMUL_FUSION_MAX_OUTPUT_SIZE: int = int(os.getenv("MATMUL_FUSION_MAX_OUTPUT_SIZE", "-1"))
"""
If there are fusible matmul siblings with the total output dimention size larger than this number,
they will not be fused by the pass `FuseQKVProjections`.
If this value is negative, all matmul siblings will be fused.
"""
