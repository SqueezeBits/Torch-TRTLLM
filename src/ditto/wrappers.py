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
        input_processors: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        output_processors: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        constant_inputs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model: InnerModuleType = model
        self.input_processors = input_processors or {}
        self.output_processors = output_processors or {}
        self.constant_inputs = constant_inputs or {}

    def preprocess(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        for _name, input_processor in self.input_processors.items():
            kwargs = input_processor(kwargs)
        return kwargs

    def postprocess(self, outputs: dict[str, Any]) -> dict[str, Any]:
        for _name, output_processor in self.output_processors.items():
            outputs = output_processor(outputs)
        return outputs

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        kwargs = self.preprocess(kwargs)
        outputs = self.model(**kwargs, **self.constant_inputs)
        assert isinstance(outputs, dict), "The tuple output is not supported. You may need to set `return_dict=True`"
        return self.postprocess({**outputs})


class PreExportWrapper(ExportWrapper[torch.nn.Module]):
    pass


class PostExportWrapper(ExportWrapper[GraphModule]):
    def preprocess(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = super().preprocess(kwargs)
        if isinstance((forward_arg_names := self.model.meta.get("forward_arg_names", None)), list):
            kwargs = {name: value for name, value in kwargs.items() if name in forward_arg_names}
        return kwargs
