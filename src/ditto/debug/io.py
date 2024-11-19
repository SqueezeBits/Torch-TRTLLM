import os
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from functools import cache
from typing import IO, Any

import onnx
from loguru import logger
from onnx.external_data_helper import _get_all_tensors, uses_external_data

from ..config import DEBUG_ARTIFACTS_DIR, DEFAULT_ONNX_PROTO_SIZE_THRESHOLD


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


def save_onnx_without_weights(
    proto: onnx.ModelProto,
    f: IO[bytes],
    *,
    size_threshold: int = DEFAULT_ONNX_PROTO_SIZE_THRESHOLD,
    convert_attribute: bool = False,
) -> None:
    onnx.convert_model_to_external_data(
        proto,
        location="null",
        size_threshold=size_threshold,
        convert_attribute=convert_attribute,
    )
    for tensor in _get_all_tensors(proto):
        if uses_external_data(tensor) and tensor.HasField("raw_data"):
            tensor.ClearField("raw_data")
    # pylint: disable-next=protected-access
    serialized_proto = onnx._get_serializer(None, f).serialize_proto(proto)
    f.write(serialized_proto)
