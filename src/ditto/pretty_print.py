import contextlib
from typing import Generator

import torch


@contextlib.contextmanager
def brief_tensor_repr() -> Generator[None, None, None]:
    def __brief_tensor_repr__(self: torch.Tensor, *, tensor_contents=None) -> str:
        dtype_repr = f"{self.dtype}".removeprefix("torch.")
        return f"Tensor(shape={tuple(self.shape)}, dtype={dtype_repr})"

    original_tensor__repr__ = torch.Tensor.__repr__
    torch.Tensor.__repr__ = __brief_tensor_repr__
    try:
        yield None
    finally:
        torch.Tensor.__repr__ = original_tensor__repr__
