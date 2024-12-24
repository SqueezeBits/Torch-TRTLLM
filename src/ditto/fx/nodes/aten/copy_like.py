# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
import torch

from ...utils import get_tensor_metadata
from ..asterick import Asterick
from .aten_op import FinalATenOp
from .unary_elementwise import UnaryElementwise


@UnaryElementwise.register(torch.ops.aten.clone.default)
class Clone(UnaryElementwise, FinalATenOp):
    asterick: None = Asterick
    memory_format: torch.memory_format | None = None


@UnaryElementwise.register(torch.ops.aten._to_copy.default)
class ToCopy(UnaryElementwise, FinalATenOp):
    """ATen op _to_copy::default.

    ```
    aten::_to_copy(
        Tensor self,
        *,
        ScalarType? dtype=None,
        Layout? layout=None,
        Device? device=None,
        bool? pin_memory=None,
        bool non_blocking=False,
        MemoryFormat? memory_format=None
    ) -> Tensor
    ```
    """

    asterick: None = Asterick
    dtype: torch.dtype | None = None
    layout: torch.layout | None = None
    device: torch.device | None = None
    pin_memory: bool | None = None
    non_blocking: bool = False
    memory_format: torch.memory_format | None = None

    @property
    def dtype_unchanged(self) -> bool:
        if self.dtype is None:
            return True
        if meta := get_tensor_metadata(self.this):
            return meta.dtype == self.dtype
        return False
