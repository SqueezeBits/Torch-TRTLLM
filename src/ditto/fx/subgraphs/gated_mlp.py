from torch.fx import Node
from typing_extensions import Self

from ..nodes import MM, MulTensorTensor
from ..utils import find_closest_common_descendant
from .fused_linear import FusedLinear
from .linear import Linear
from .path import TrailingReformatPath
from .subgraph import Subgraph


class GatedMLP(Subgraph):
    """A subgraph representing a gated MLP layer.

    Attributes:
        up_proj (Linear): The up projection linear layer
        gate_proj (Linear): The gate projection linear layer
        mul (MulTensorTensor): The multiplication of the up projection and gate projection
    """

    up_proj: Linear
    gate_proj: Linear
    mul: MulTensorTensor

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (mul := MulTensorTensor.specialize_from(node)):
            return None

        this_top = TrailingReformatPath.configure_from(mul.this).top
        other_top = TrailingReformatPath.configure_from(mul.other).top
        if up_proj := Linear.configure_from(this_top):
            gate_proj = Linear.find_nearest(other_top)
        elif up_proj := Linear.configure_from(other_top):
            gate_proj = Linear.find_nearest(this_top)
        elif (
            (fused_proj := FusedLinear.find_nearest(this_top, break_if=lambda n: MM.specialize_from(n) is not None))
            and len(fused_proj) == 2
            and find_closest_common_descendant(
                fused_proj.slices[0].node,
                fused_proj.slices[1].node,
            )
            == mul.node
        ):
            up_proj = gate_proj = fused_proj.linear
        else:
            return None

        if not (
            gate_proj and up_proj.input_node == gate_proj.input_node and up_proj.add is None and gate_proj.add is None
        ):
            return None

        return cls(up_proj=up_proj, gate_proj=gate_proj, mul=mul)
