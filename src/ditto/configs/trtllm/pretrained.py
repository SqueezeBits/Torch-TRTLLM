from pydantic import Field

from ...types import StrictlyTyped
from .literals import DTypeLiteral, QuantAlgoLiteral


class TRTLLMMapping(StrictlyTyped):
    """Minimal set of properties for initializing `trtllm.Mapping`."""

    world_size: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=8, ge=1)
    cp_size: int = Field(default=1, ge=1)
    tp_size: int = Field(default=1, ge=1)
    pp_size: int = Field(default=1, ge=1)
    moe_tp_size: int = Field(default=1, ge=1)
    moe_ep_size: int = Field(default=1, ge=1)


class TRTLLMQuantConfig(StrictlyTyped):
    quant_algo: QuantAlgoLiteral | None = None
    kv_cache_quant_algo: QuantAlgoLiteral | None = None
    group_size: int = 128
    smoothquant_val: float = 0.5
    """Only required for SmoothQuant"""
    clamp_val: list[float] | None = Field(default=None, min_length=2, max_length=2)
    """Only required for SmoothQuant"""
    has_zero_point: bool = False
    pre_quant_scale: bool = False
    exclude_modules: list[str] | None = None


class TRTLLMPretrainedConfig(StrictlyTyped):
    """Minimal subset of properties in `trtllm.PretrainedConfig` required at runtime."""

    architecture: str
    dtype: DTypeLiteral = "float16"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    mapping: TRTLLMMapping = Field(default_factory=TRTLLMMapping)
    quantization: TRTLLMQuantConfig | None = None
