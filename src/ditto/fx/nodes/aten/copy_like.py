# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

from abc import abstractmethod
from typing import Literal

import torch

from ...utils import get_tensor_metadata
from ..asterick import Asterick
from .unary_elementwise import UnaryElementwise


class CopyLike(UnaryElementwise):
    asterick: None = Asterick

    @property
    @abstractmethod
    def is_pure_copy(self) -> bool:
        ...


@CopyLike.final(torch.ops.aten.clone.default)
class Clone(CopyLike):
    memory_format: torch.memory_format | None = None

    @property
    def is_pure_copy(self) -> Literal[True]:
        return True


@CopyLike.final(torch.ops.aten._to_copy.default)
class ToCopy(CopyLike):
    dtype: torch.dtype | None = None
    layout: torch.layout | None = None
    device: torch.device | None = None
    pin_memory: bool | None = None
    non_blocking: bool = False
    memory_format: torch.memory_format | None = None

    @property
    def is_pure_copy(self) -> bool:
        if meta := get_tensor_metadata(self.this):
            return meta.dtype == self.dtype
        return False
