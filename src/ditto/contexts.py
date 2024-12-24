import contextlib
import logging
from collections.abc import Generator
from typing import Any

import torch
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import log as symbolic_shape_logger


@contextlib.contextmanager
def brief_tensor_repr() -> Generator[None, None, None]:
    def tensor_repr(self: torch.Tensor, *, _: Any = None) -> str:
        dtype_repr = f"{self.dtype}".removeprefix("torch.")
        return f"Tensor(shape={tuple(self.shape)}, dtype={dtype_repr}, device={self.device})"

    original_tensor__repr__ = torch.Tensor.__repr__
    torch.Tensor.__repr__ = tensor_repr  # type: ignore[method-assign, assignment]
    try:
        yield None
    finally:
        torch.Tensor.__repr__ = original_tensor__repr__  # type: ignore[method-assign, assignment]


@contextlib.contextmanager
def detailed_sym_node_str() -> Generator[None, None, None]:
    def sym_node_str(self: SymNode) -> str:
        if self._expr != self.expr:
            return f"{self._expr}({self.expr})"
        return f"{self.expr}"

    original_sym_node_str = SymNode.str
    SymNode.str = sym_node_str  # type: ignore[method-assign]
    try:
        yield None
    finally:
        SymNode.str = original_sym_node_str  # type: ignore[method-assign]


@contextlib.contextmanager
def ignore_symbolic_shapes_warning() -> Generator[None, None, None]:
    log_level = symbolic_shape_logger.level
    symbolic_shape_logger.setLevel(logging.ERROR)
    try:
        yield None
    finally:
        symbolic_shape_logger.setLevel(log_level)
