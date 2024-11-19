import os
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from functools import cache
from typing import IO, Any

import onnx
from loguru import logger

from ..config import DEBUG_ARTIFACTS_DIR


@contextmanager
def none_context() -> Generator[None, None, None]:
    try:
        yield None
    finally:
        pass


@cache
def get_debug_artifacts_dir() -> str | None:
    if DEBUG_ARTIFACTS_DIR is None:
        return None
    os.makedirs(DEBUG_ARTIFACTS_DIR, exist_ok=True)
    logger.info(f"DEBUG_ARTIFACTS_DIR: {DEBUG_ARTIFACTS_DIR}")
    return DEBUG_ARTIFACTS_DIR


def open_debug_artifact(filename: str, mode: str = "w") -> IO[Any] | nullcontext[None]:
    if d := get_debug_artifacts_dir():
        path = os.path.join(d, filename)
        actions = {
            "w": "Writing",
            "a": "Appending",
            "r": "Reading",
        }
        for k, v in actions.items():
            if k in mode:
                action = v
                break
        else:
            action = "Opening"
        logger.info(f"{action} debug artifact at {path}")
        return open(path, mode)
    return nullcontext(None)


def save_onnx_without_weights(proto: onnx.ModelProto, f: IO[bytes] | str) -> None:
    if isinstance(f, str):
        path = f
    else:
        path = f.name
    out_dir = os.path.dirname(path)
    file = os.path.basename(path)
    filename, _ = os.path.splitext(file)
    weight_file = f"{filename}.bin"
    onnx.save(proto, path, save_as_external_data=True, location=weight_file)
    if os.path.isfile(weight_path := os.path.join(out_dir, weight_file)):
        os.remove(weight_path)
