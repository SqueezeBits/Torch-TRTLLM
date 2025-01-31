# pyright: reportPrivateImportUsage=false
from collections.abc import Sequence

from loguru import logger
from peft import PEFT_TYPE_TO_CONFIG_MAPPING, LoraConfig, PeftConfig, PeftModel
from transformers import PreTrainedModel

from .constants import PEFT_ADAPTER_PREFIX


def load_peft_adapters(model: PreTrainedModel, peft_ids: Sequence[str]) -> PreTrainedModel | PeftModel:
    """Load PEFT adapters onto a model.

    Args:
        model (PreTrainedModel): Base model to load adapters onto
        peft_ids (Sequence[str]): Sequence of PEFT adapter IDs to load

    Returns:
        PreTrainedModel | PeftModel: Model with PEFT adapters loaded, or original model if no adapters
    """
    peft_model: PeftModel | None = None
    for task_uid, peft_id in sorted_enumerate(peft_ids):
        logger.info(f"Loading PEFT adapter {peft_id} with {task_uid=}")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, peft_id, adapter_name=f"{PEFT_ADAPTER_PREFIX}_{task_uid}")
        else:
            _ = peft_model.load_adapter(peft_id, adapter_name=f"{PEFT_ADAPTER_PREFIX}_{task_uid}")
    if peft_model is None:
        return model
    peft_model.base_model.set_adapter([f"{PEFT_ADAPTER_PREFIX}_{i}" for i in range(len(peft_ids))])
    return peft_model


def sorted_enumerate(peft_ids: Sequence[str]) -> list[tuple[int, str]]:
    """Enumerate PEFT IDs sorted by LoRA rank.

    Note that the adapters need to be loaded in increasing order of LoRA rank.
    Otherwise, some of the LoRA weights can be contaminated.

    Args:
        peft_ids (Sequence[str]): Sequence of PEFT adapter IDs

    Returns:
        list[tuple[int, str]]: List of (index, peft_id) tuples sorted by LoRA rank
    """
    return sorted(enumerate(peft_ids), key=lambda x: load_lora_config(x[1]).r)


def load_lora_config(peft_id: str) -> LoraConfig:
    """Load LoRA configuration from a PEFT adapter ID.

    Args:
        peft_id (str): PEFT adapter ID to load config from

    Returns:
        LoraConfig: LoRA configuration

    Raises:
        ValueError: If the PEFT adapter is not a LoRA adapter
    """
    config = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig._get_peft_type(peft_id)].from_pretrained(peft_id)
    if not isinstance(config, LoraConfig):
        raise ValueError(f"Unsupported PEFT type: {peft_id}. Currently only Lora is supported.")
    return config
