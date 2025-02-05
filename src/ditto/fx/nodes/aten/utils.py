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

from loguru import logger


def make_dim_nonnegative(dim: int, *, ndim: int) -> int:
    """Make a dimension nonnegative.

    Args:
        dim (int): The dimension to make nonnegative.
        ndim (int): The number of dimensions of the tensor.

    Returns:
        int: The nonnegative dimension.
    """
    if not -ndim <= dim < ndim:
        logger.warning(f"dimension out of range: expected dim={dim} to in range({-ndim}, {ndim})")
    return dim if dim >= 0 else dim + ndim


def make_axis_nonnegative(axis: int, *, dim_size: int) -> int:
    """Make an axis nonnegative.

    Args:
        axis (int): The axis to make nonnegative.
        dim_size (int): The size of the dimension.

    Returns:
        int: The nonnegative axis.
    """
    if not -dim_size <= axis <= dim_size:
        logger.warning(f"axis out of range: expected axis={axis} to in range({-dim_size}, {dim_size})")
    return axis if axis >= 0 else axis + dim_size


def has_same_values(x: int | None, y: int | None) -> bool:
    """Check if two values are the same.

    Args:
        x (int | None): The first value.
        y (int | None): The second value.

    Returns:
        bool: True if the values are the same, False otherwise.
    """
    if x is None or y is None:
        return False
    return x == y
