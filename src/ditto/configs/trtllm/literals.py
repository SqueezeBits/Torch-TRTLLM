from typing import Literal

# The available data types are listed in `_str_to_trt_dtype_dict` from `tensorrt_llm._utils`
# which is used by the function `str_dtype_to_trt` in the same file.
DTypeLiteral = Literal[
    "float16",
    "bfloat16",
    "float32",
    "int8",
    "int32",
    "int64",
    "bool",
    "fp8",
]

KVCacheTypeLiteral = Literal["CONTINUOUS", "DISABLED", "PAGED"]

LoraCheckpointLiteral = Literal["hf", "nemo"]

PluginFlag = Literal["auto", "float16", "bfloat16", "fp8", None]

QuantAlgoLiteral = Literal[
    "W8A16",
    "W4A16",
    "W4A16_AWQ",
    "W4A8_AWQ",
    "W4A16_GPTQ",
    "W8A8_SQ_PER_CHANNEL",
    "W8A8_SQ_PER_TENSOR_PLUGIN",
    "W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN",
    "W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN",
    "W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN",
    "FP8",
    "FP8_PER_CHANNEL_PER_TOKEN",
    "INT8",
]
