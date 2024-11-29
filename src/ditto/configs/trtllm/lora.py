from pydantic import Field

from ...types import StrictlyTyped
from .literals import LoraCheckpointLiteral


class TRTLLMLoraConfig(StrictlyTyped):
    lora_dir: list[str] = Field(default_factory=list)
    lora_ckpt_source: LoraCheckpointLiteral = "hf"
    max_lora_rank: int = 64
    lora_target_modules: list[str] = Field(default_factory=list)
    trtllm_modules_to_hf_modules: dict[str, str] = Field(default_factory=dict)
