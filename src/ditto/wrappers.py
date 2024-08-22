from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput

from .cache_handler import CacheHandler


class ExportWrapper(torch.nn.Module, ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        cache_handler: CacheHandler,
        input_ids_key: str = "input_ids",
        attention_mask_key: str = "attention_mask",
        past_key_values_key: str = "past_key_values",
        seq_dim: int = -1,
    ) -> None:
        super().__init__()
        self.model = model
        self.cache_handler = cache_handler
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        self.past_key_values_key = past_key_values_key
        self.seq_dim = seq_dim

    @abstractmethod
    def preprocess(self, kwargs: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def postprocess(self, output: ModelOutput) -> None:
        ...

    def forward(self, **kwargs: Any) -> ModelOutput:
        self.preprocess(kwargs)
        output = self.model(**kwargs)
        assert isinstance(output, ModelOutput), (
            "The tuple output is not supported. You may need to set `return_dict=True`"
        )
        self.postprocess(output)
        return output


class PreExportWrapper(ExportWrapper):
    def preprocess(self, kwargs: dict[str, Any]) -> None:
        if not isinstance((past_key_values := kwargs.get(self.past_key_values_key)), torch.Tensor):
            raise ValueError(f"Expected {self.past_key_values_key} to be a tensor but got {past_key_values}")
        kwargs[self.past_key_values_key] = self.cache_handler.to_cache(past_key_values)

        if not self.cache_handler.is_static:
            if not isinstance((prefilled_attention_mask := kwargs.pop("prefilled_attention_mask", None)), torch.Tensor):
                raise ValueError(f"Expected prefilled_attention_mask to be a tensor but got {prefilled_attention_mask}")
            if not isinstance((generation_attention_mask := kwargs.pop("generation_attention_mask", None)), torch.Tensor):
                raise ValueError(f"Expected generation_attention_mask to be a tensor but got {generation_attention_mask}")
            kwargs[self.attention_mask_key] = torch.cat(
                (prefilled_attention_mask, generation_attention_mask),
                dim=self.seq_dim,
            )

    def postprocess(self, output: ModelOutput) -> None:
        if not isinstance((past_key_values := getattr(output, self.past_key_values_key, None)), Cache):
            raise ValueError(f"Expected {self.past_key_values_key} to be a cache but got {past_key_values}")
        output.past_key_values = self.cache_handler.to_tensor(past_key_values)


class PostExportWrapper(ExportWrapper):
    def preprocess(self, kwargs: dict[str, Any]) -> None:
        if not isinstance((past_key_values := kwargs.get(self.past_key_values_key)), Cache):
            raise ValueError(f"Expected {self.past_key_values_key} to be a cache but got {past_key_values}")
        past_key_values_tensor = self.cache_handler.to_tensor(past_key_values)
        kwargs[self.past_key_values_key] = past_key_values_tensor

        if not self.cache_handler.is_static:
            if not isinstance((input_ids := kwargs.get(self.input_ids_key)), torch.Tensor):
                raise ValueError(f"Expected {self.input_ids_key} to be a tensor but got {input_ids}")
            if not isinstance((attention_mask := kwargs.pop(self.attention_mask_key, None)), torch.Tensor):
                raise ValueError(f"Expected {self.attention_mask_key} to be a tensor but got {attention_mask}")
            input_ids_len = input_ids.shape[self.seq_dim]
            cache_len = past_key_values_tensor.shape[-2]
            assert attention_mask.shape[self.seq_dim] == input_ids_len + cache_len
            prefilled_attention_mask, generation_attention_mask = torch.split(
                attention_mask, [cache_len, input_ids_len], dim=self.seq_dim
            )
            kwargs["prefilled_attention_mask"] = prefilled_attention_mask
            kwargs["generation_attention_mask"] = generation_attention_mask

    def postprocess(self, output: ModelOutput) -> None:
        assert isinstance(output, ModelOutput), (
            "The tuple output is not supported. You may need to set `return_dict=True`"
        )
        if not isinstance((past_key_values := getattr(output, self.past_key_values_key, None)), torch.Tensor):
            raise ValueError(f"Expected {self.past_key_values_key} to be a tensor but got {past_key_values}")
        output.past_key_values = self.cache_handler.to_cache(past_key_values)
