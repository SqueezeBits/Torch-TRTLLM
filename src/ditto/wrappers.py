from collections.abc import Callable
from typing import Any, Generic, TypeVar

import torch
from torch.fx import GraphModule

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
