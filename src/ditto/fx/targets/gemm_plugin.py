from enum import IntEnum, IntFlag
from typing import Any

import numpy as np
import tensorrt as trt
import torch

from ...types import StrictlyTyped
from .plugin_field_types import PLUGIN_FIELD_TYPES


class GemmPluginFields(StrictlyTyped):
    # the order of the attributes does matter!
    transa: int = 0
    transb: int = 0
    pad_lda: int = 0
    pad_ldb: int = 0
    pad_ldc: int = 0
    type_id: trt.DataType
    use_fp8: int = 0
    alpha: float = 1.0

    def get_plugin_fields(self) -> list[trt.PluginField]:
        def convert_to_plugin_field(name: str, value: Any) -> trt.PluginField:
            dtype: type[np.number]
            if name == "alpha":
                dtype = np.float32
            else:
                dtype = np.int32
            plugin_field_type = PLUGIN_FIELD_TYPES[dtype]

            if isinstance(value, IntEnum | IntFlag | trt.DataType):
                value = value.value
            return trt.PluginField(name, np.array(value, dtype=dtype), plugin_field_type)

        return [convert_to_plugin_field(name, value) for name, value in self.model_dump().items()]


class GemmPlugin(GemmPluginFields):
    @property
    def __name__(self) -> str:
        return "gemm_plugin"

    def __hash__(self) -> int:
        return hash(f"gemm_plugin_{id(self)}")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GemmPlugin):
            return self is other
        return False

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return x @ weight.transpose(1, 0)
