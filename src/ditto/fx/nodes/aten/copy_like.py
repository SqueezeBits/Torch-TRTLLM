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

# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
import torch

from ...utils import get_tensor_metadata
from ..asterisk import Asterisk
from .aten_op import FinalATenOp
from .unary_elementwise import UnaryElementwise


@UnaryElementwise.register(torch.ops.aten.clone.default)
class Clone(UnaryElementwise, FinalATenOp):
    """Specialization for the clone operator.

    Attributes:
        asterisk (None): The asterisk of the clone.
        memory_format (torch.memory_format | None): The memory format of the clone.
    """

    asterisk: None = Asterisk
    memory_format: torch.memory_format | None = None


@UnaryElementwise.register(torch.ops.aten._to_copy.default)
class ToCopy(UnaryElementwise, FinalATenOp):
    """Specialization for the _to_copy operator.

    Attributes:
        asterisk (None): The asterisk of the _to_copy.
        dtype (torch.dtype | None): The dtype of the _to_copy.
        layout (torch.layout | None): The layout of the _to_copy.
        device (torch.device | None): The device of the _to_copy.
        pin_memory (bool | None): Whether the _to_copy is pinned in memory.
        non_blocking (bool): Whether the _to_copy is non-blocking.
        memory_format (torch.memory_format | None): The memory format of the _to_copy.
    """

    asterisk: None = Asterisk
    dtype: torch.dtype | None = None
    layout: torch.layout | None = None
    device: torch.device | None = None
    pin_memory: bool | None = None
    non_blocking: bool = False
    memory_format: torch.memory_format | None = None

    @property
    def dtype_unchanged(self) -> bool:
        """Check if the dtype is unchanged.

        Returns:
            bool: True if the dtype is unchanged, False otherwise.
        """
        if self.dtype is None:
            return True
        if meta := get_tensor_metadata(self.this):
            return meta.dtype == self.dtype
        return False
