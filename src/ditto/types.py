# pylint: disable=unused-import
from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict
from torch._C import _SDPBackend as SDPBackend  # noqa: F401
from torch.export.dynamic_shapes import _Dim as ExportDim  # noqa: F401

BuiltInConstant = int | float | bool | None
DeviceLikeType = str | torch.device | int
Number = int | float | bool
SymInt = int | torch.SymInt
SymbolicShape = tuple[SymInt, ...]


class StrictlyTyped(BaseModel):
    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "validate_default": True,
    }
