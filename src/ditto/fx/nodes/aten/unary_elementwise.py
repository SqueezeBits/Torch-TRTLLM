# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch

from ..asterick import Asterick
from .unary import Unary


class UnaryElementwise(Unary):
    ...


@UnaryElementwise.final(torch.ops.aten.clone.default)
class Clone(UnaryElementwise):
    asterick: None = Asterick
    memory_format: torch.memory_format | None = None


@UnaryElementwise.final(torch.ops.aten.sigmoid.default)
class Sigmoid(UnaryElementwise):
    ...


@UnaryElementwise.final(torch.ops.aten.sqrt.default)
class Sqrt(UnaryElementwise):
    ...


@UnaryElementwise.final(torch.ops.aten._to_copy.default)
class ToCopy(UnaryElementwise):
    asterick: None = Asterick
    dtype: torch.dtype | None = None
    layout: torch.layout | None = None
    device: torch.device | None = None
    pin_memory: bool | None = None
    non_blocking: bool = False
    memory_format: torch.memory_format | None = None
