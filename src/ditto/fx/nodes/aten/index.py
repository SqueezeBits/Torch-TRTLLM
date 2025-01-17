# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import torch
from torch.fx.node import Node

from ...nodes.get_attr import GetAttr
from .aten_op import ATenOp, FinalATenOp


@ATenOp.register(torch.ops.aten.index.Tensor)
class Index(FinalATenOp):
    this: Node
    indices: tuple[Node | None, ...]

    @property
    def can_replace_with_single_slice(self) -> bool:
        return (
            len(self.node.all_input_nodes) == 2
            and (constant := GetAttr.specialize_from(self.node.all_input_nodes[1]))
            and isinstance(constant.parameter, torch.Tensor)
            and constant.parameter.numel() == 1
        )

    @property
    def dim(self) -> int:
        assert self.can_replace_with_single_slice
        return self.indices.index(self.node.all_input_nodes[-1])

    @property
    def idx(self) -> int:
        assert self.can_replace_with_single_slice

        idx_tensor = GetAttr.specialize_from(self.node.all_input_nodes[-1]).parameter
        idx = int(idx_tensor.item())

        if idx < 0 and "val" in self.this.meta and isinstance(dim_size := self.this.meta["val"].size(self.dim), int):
            idx = dim_size + idx

        return idx
