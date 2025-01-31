from pydantic import Field

from ...literals import LoraCheckpointLiteral, LoraPluginInputPrefix
from ...types import StrictlyTyped


class TRTLLMLoraConfig(StrictlyTyped):
    """LoRA configuration for TRT-LLM.

    Attributes:
        lora_dir: List of directories containing LoRA checkpoints.
        lora_ckpt_source: Source format of LoRA checkpoints ('hf' for HuggingFace, 'nemo' for Nemo).
        max_lora_rank: Maximum rank for LoRA adapters.
        lora_target_modules: List of module prefixes to apply LoRA to.
        trtllm_modules_to_hf_modules: Mapping from TRT-LLM module prefixes to HuggingFace module names.
    """

    lora_dir: list[str] = Field(default_factory=list)
    lora_ckpt_source: LoraCheckpointLiteral = "hf"
    max_lora_rank: int = 64
    lora_target_modules: list[LoraPluginInputPrefix] = Field(default_factory=list)
    trtllm_modules_to_hf_modules: dict[LoraPluginInputPrefix, str] = Field(default_factory=dict)
