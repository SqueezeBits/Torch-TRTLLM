# pylint: disable=unused-argument
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import torch
from torch.fx import GraphModule
from transformers import PreTrainedModel

# pylint: disable-next=invalid-name
InnerModuleType = TypeVar("InnerModuleType", torch.nn.Module, GraphModule)


class ExportWrapper(torch.nn.Module, Generic[InnerModuleType]):
    def __init__(
        self,
        model: InnerModuleType,
        *,
        process_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        process_outputs: Callable[[Any], Any] | None = None,
        constant_inputs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model: InnerModuleType = model
        self.process_inputs = process_inputs
        self.process_outputs = process_outputs
        self.constant_inputs = constant_inputs or {}

    def preprocess(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.process_inputs:
            return self.process_inputs(kwargs)
        return kwargs

    def postprocess(self, outputs: Any) -> Any:
        if self.process_outputs:
            return self.process_outputs(outputs)
        return outputs

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        kwargs = self.preprocess(kwargs)
        outputs = self.model(**kwargs, **self.constant_inputs)
        return self.postprocess(outputs)


class PreExportWrapper(ExportWrapper[torch.nn.Module]):
    pass


class PostExportWrapper(ExportWrapper[GraphModule]):
    def preprocess(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = super().preprocess(kwargs)
        if isinstance((forward_arg_names := self.model.meta.get("forward_arg_names", None)), list):
            kwargs = {name: value for name, value in kwargs.items() if name in forward_arg_names}
        return kwargs


class TRTLLMPreTrainedModelWrapper(ExportWrapper[PreTrainedModel]):
    # pylint: disable-next=arguments-differ
    def forward(  # type: ignore[override]
        self,
        *,
        input_ids: torch.Tensor,
        last_token_ids: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        sequence_length: torch.Tensor,
        host_request_types: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        context_lengths: torch.Tensor,
        host_runtime_perf_knobs: torch.Tensor,
        host_context_lengths: torch.Tensor,
        host_max_attention_window_sizes: torch.Tensor,
        host_sink_token_length: torch.Tensor,
        cache_indirection: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().forward(input_ids=input_ids, **kwargs)
