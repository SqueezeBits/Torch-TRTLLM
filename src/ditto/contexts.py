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
import logging
import operator
import random
from collections.abc import Callable, Generator
from functools import reduce
from hashlib import md5
from typing import Any

import torch
import torch.jit._state as torch_jit_state
from loguru import logger
from peft import PeftModel
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import log as symbolic_shape_logger

from .literals import LogLevelLiteral
from .types import verify


@contextlib.contextmanager
def brief_tensor_repr() -> Generator[None, None, None]:
    """Context manager that temporarily modifies tensor representation to be more concise.

    While active, tensor.__repr__() will only show shape, dtype and device information.
    """

    def tensor_repr(self: torch.Tensor, *, _: Any = None) -> str:
        dtype_repr = f"{self.dtype}".removeprefix("torch.")
        return f"Tensor(shape={tuple(self.shape)}, dtype={dtype_repr}, device={self.device})"

    original_tensor__repr__ = torch.Tensor.__repr__
    torch.Tensor.__repr__ = tensor_repr
    try:
        yield None
    finally:
        torch.Tensor.__repr__ = original_tensor__repr__


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
    SymNode.str = sym_node_str
    try:
        yield None
    finally:
        SymNode.str = original_sym_node_str


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
    modelopt_cache = verify(getattr(PeftModel, "_modelopt_cache", None), as_type=dict[str, Callable[..., Any]]) or {}
    is_peft_patched_by_modelopt = len(modelopt_cache) > 0
    modelopt_patched_methods: dict[str, Callable[..., Any]] = {}
    try:
        if is_peft_patched_by_modelopt:
            for method_name, original_method in modelopt_cache.items():
                if not hasattr(PeftModel, method_name):
                    continue
                modelopt_patched_methods[method_name] = getattr(PeftModel, method_name)
                setattr(PeftModel, method_name, original_method)
                logger.debug(f"modelopt patches for {PeftModel.__name__}.{method_name} disabled")
        yield None
    finally:
        if is_peft_patched_by_modelopt:
            for method_name, patched_method in modelopt_patched_methods.items():
                setattr(PeftModel, method_name, patched_method)
                logger.debug(f"modelopt patches for {PeftModel.__name__}.{method_name} enabled")


@contextlib.contextmanager
def temporary_random_seed(*seeds: int | str | bytes | None) -> Generator[None, None, None]:
    """Context manager that temporarily sets a random seed.

    While active, sets a compound random seed derived from the input seeds.
    The original random state is restored on exit.

    Args:
        *seeds (int | str | bytes | None): Variable number of seed values that can be integers, strings, bytes or None.
               These are combined into a single compound seed using XOR. If no seeds are provided, the random state
               remains unchanged.
    """

    def to_int(value: int | str | bytes | None) -> int:
        b: bytes
        if isinstance(value, int):
            b = value.to_bytes(length=16, byteorder="big")
        elif isinstance(value, str):
            b = value.encode()
        elif value is None:
            b = b""
        else:
            b = value
        return int(md5(b).hexdigest(), 16)

    original_seed = random.getstate()
    compound_seed = reduce(operator.xor, (to_int(seed) for seed in seeds)) if seeds else None
    if compound_seed is not None:
        random.seed(compound_seed)
    try:
        yield None
    finally:
        random.setstate(original_seed)


@contextlib.contextmanager
def set_logger_level(logger_name: str, log_level: LogLevelLiteral) -> Generator[None, None, None]:
    """Context manager that temporarily sets a logger's level.

    While active, sets the specified logger to the given log level.
    The original log level is restored on exit.

    Args:
        logger_name (str): The name of the logger to modify
        log_level (int): The log level to temporarily set.
    """
    internal_logger = logging.getLogger(logger_name)
    original_level = internal_logger.level
    internal_logger.setLevel(log_level)
    try:
        yield None
    finally:
        internal_logger.setLevel(original_level)
