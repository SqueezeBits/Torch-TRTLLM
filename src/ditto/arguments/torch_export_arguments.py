import torch
from pydantic import Field
from typing_extensions import Self

from ..constants import DEFAULT_DEVICE
from ..contexts import brief_tensor_repr
from ..types import BuiltInConstant, DeviceLikeType, ExportDim, StrictlyTyped
from .dynamic_dim import DynamicDimensionType
from .tensor_type_hint import TensorTypeHint


class TorchExportArguments(StrictlyTyped):
    tensor_inputs: dict[str, torch.Tensor]
    constant_inputs: dict[str, BuiltInConstant] = Field(default_factory=dict)
    constraints: dict[str, dict[int, ExportDim] | None] = Field(default_factory=dict)

    def __str__(self) -> str:
        def dim_repr(dim: ExportDim) -> str:
            min_ = getattr(dim, "min", "?")
            max_ = getattr(dim, "max", "?")
            return f"{dim.__name__}(min={min_}, max={max_})"

        lines: list[str] = [type(self).__name__]
        with brief_tensor_repr():
            lines.append("=============Tensor inputs for torch.export=============")
            for name, tensor in self.tensor_inputs.items():
                lines.append(f"{name}: {tensor}")
            lines.append("============Constant inputs for torch.export============")
            for name, constant in self.constant_inputs.items():
                lines.append(f"{name}: {constant}")
            lines.append("======================Constraints=======================")
            for name, constraint in self.constraints.items():
                if constraint is None:
                    lines.append(f"{name}: None")
                    continue
                dim_reprs = {axis: dim_repr(dim) for axis, dim in constraint.items()}
                lines.append(f"{name}: {dim_reprs}")
            lines.append("========================================================")
        return "\n".join(lines)

    @classmethod
    def from_hints(
        cls,
        *,
        device: DeviceLikeType = DEFAULT_DEVICE,
        **input_hints: TensorTypeHint | BuiltInConstant,
    ) -> Self:
        tensor_inputs: dict[str, torch.Tensor] = {}
        constant_inputs: dict[str, BuiltInConstant] = {}
        constraints: dict[str, dict[int, ExportDim] | None] = {}

        for name, hint in input_hints.items():
            if isinstance(hint, BuiltInConstant):
                constant_inputs[name] = hint
                continue
            shape = tuple(s if isinstance(s, int) else s.example for s in hint.shape)
            tensor_inputs[name] = torch.zeros(*shape, dtype=hint.dtype, device=device)

            for dim, s in enumerate(hint.shape):
                if name not in constraints:
                    constraints[name] = {}
                if not isinstance(s, DynamicDimensionType):
                    continue
                assert (constraint := constraints[name]) is not None
                if not isinstance(export_dim := s.export_dim, int):
                    constraint[dim] = export_dim

            if not constraints[name]:
                constraints[name] = None

        return cls(
            tensor_inputs=tensor_inputs,
            constant_inputs=constant_inputs,
            constraints=constraints,
        )
