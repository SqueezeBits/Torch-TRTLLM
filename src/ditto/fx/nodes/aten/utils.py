from loguru import logger


def make_dim_nonnegative(dim: int, *, ndim: int) -> int:
    if not -ndim <= dim < ndim:
        logger.warning(f"dimension out of range: expected dim={dim} to in range({-ndim}, {ndim})")
    return dim if dim >= 0 else dim + ndim


def make_axis_nonnegative(axis: int, *, dim_size: int) -> int:
    if not -dim_size <= axis <= dim_size:
        logger.warning(f"axis out of range: expected axis={axis} to in range({-dim_size}, {dim_size})")
    return axis if axis >= 0 else axis + dim_size


def has_same_values(x: int | None, y: int | None) -> bool:
    if x is None or y is None:
        return False
    return x == y
