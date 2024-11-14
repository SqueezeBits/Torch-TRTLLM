import io
import os
from functools import cache
from typing import IO, Literal

from loguru import logger

from .config import DEBUG_ARTIFACTS_DIR


def make_dim_nonnegative(dim: int, *, ndim: int) -> int:
    if not -ndim <= dim < ndim:
        logger.warning(f"dimension out of range: expected dim={dim} to in range({-ndim}, {ndim})")
    return dim if dim >= 0 else dim + ndim


def make_axis_nonnegative(axis: int, *, dim_size: int) -> int:
    if not -dim_size <= axis <= dim_size:
        logger.warning(f"axis out of range: expected axis={axis} to in range({-dim_size}, {dim_size})")
    return axis if axis >= 0 else axis + dim_size


class DiscardBytesIO(io.BytesIO):
    def write(self, b) -> int:
        # Ignore the data, effectively discarding it
        return len(b)

    def getvalue(self) -> Literal[b""]:
        # Return an empty bytes object, since nothing is actually stored
        return b""


@cache
def get_debug_artifacts_dir() -> str | None:
    if DEBUG_ARTIFACTS_DIR is None:
        return None
    os.makedirs(DEBUG_ARTIFACTS_DIR, exist_ok=True)
    return DEBUG_ARTIFACTS_DIR


def open_debug_artifact(filename: str, mode: str = "w") -> IO:
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
    return DiscardBytesIO()
