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

import operator
from collections.abc import Callable
from typing import Any

from torch.fx import Node

from .call_function import FinalCallFunction


class GetItem(FinalCallFunction):
    """A specialization representing operator.getitem() nodes.

    Attributes:
        this (Node): The input sequence/container node to get item from
        idx (int): The index to retrieve
    """

    this: Node
    idx: int

    @classmethod
    def possible_targets(cls) -> tuple[Callable[..., Any], ...]:
        """Get the possible target functions for getitem operation."""
        return (operator.getitem,)
