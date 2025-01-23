# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import inspect
import logging
import sys
from collections.abc import Generator
from typing import Any

import torch
import torch.jit._state as torch_jit_state
from loguru import logger
from peft import PeftModel
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import log as symbolic_shape_logger


@contextlib.contextmanager
def brief_tensor_repr() -> Generator[None, None, None]:
    """Context manager that temporarily modifies tensor representation to be more concise.

    While active, tensor.__repr__() will only show shape, dtype and device information.
    """

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
    """Context manager that temporarily modifies SymNode string representation to show more details.

    While active, SymNode.str() will show both _expr and expr if they differ.
    """

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
    """Context manager that temporarily suppresses symbolic shape warnings.

    While active, symbolic shape warnings will be set to ERROR level.
    """
    log_level = symbolic_shape_logger.level
    symbolic_shape_logger.setLevel(logging.ERROR)
    try:
        yield None
    finally:
        symbolic_shape_logger.setLevel(log_level)


@contextlib.contextmanager
def disable_torch_jit_state() -> Generator[None, None, None]:
    """Context manager that temporarily disables PyTorch JIT scripting.

    While active, torch.jit.script will be disabled. The original state is restored on exit.
    """
    if was_enabled := torch_jit_state._enabled.enabled:
        torch_jit_state.disable()
        logger.debug("torch.jit.script disabled")
    try:
        yield None
    finally:
        if was_enabled:
            torch_jit_state.enable()
            logger.debug("torch.jit.script enabled")


@contextlib.contextmanager
def disable_modelopt_peft_patches() -> Generator[None, None, None]:
    """Context manager that temporarily disables modelopt patches to PEFT.

    While active, any modelopt patches to PEFT will be disabled by unimporting relevant modules.
    The patches are restored on exit by allowing reimport.
    """
    is_peft_patched_by_modelopt = any(m.startswith("modelopt") for m in sys.modules) and inspect.getsourcefile(
        PeftModel.load_adapter
    ) != inspect.getsourcefile(PeftModel)

    try:
        if is_peft_patched_by_modelopt:
            modules_to_unimport = {m for m in sys.modules if m.startswith("modelopt") or m.startswith("peft")}
            for m in modules_to_unimport:
                del sys.modules[m]

            logger.debug("modelopt peft patches disabled")
        yield None
    finally:
        if is_peft_patched_by_modelopt:
            logger.debug("modelopt peft patches enabled")
