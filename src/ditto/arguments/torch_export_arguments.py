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

import torch
from pydantic import Field
from typing_extensions import Self

from ..constants import DEFAULT_DEVICE
from ..contexts import brief_tensor_repr
from ..types import BuiltInConstant, DeviceLikeType, ExportDim, StrictlyTyped
from .dynamic_dim import DynamicDimensionType
from .tensor_type_hint import TensorTypeHint


class TorchExportArguments(StrictlyTyped):
    """Arguments for torch.export.

    Args:
        tensor_inputs (dict[str, torch.Tensor]): The tensor inputs for torch.export.
        constant_inputs (dict[str, BuiltInConstant]): The constant inputs for torch.export.
        constraints (dict[str, dict[int, ExportDim] | None]): The constraints for the tensor inputs.
    """

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
        """Create a TorchExportArguments object from a dictionary of input hints.

        Args:
            device (DeviceLikeType): The device to use for the tensor inputs. Defaults to `DEFAULT_DEVICE`.
            **input_hints (TensorTypeHint | BuiltInConstant): The input hints for the torch.export.

        Returns:
            Self: The TorchExportArguments object.
        """
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
                    # Note: The reason for multiplying by 2 is to satisfy the constraint
                    # imposed by some layers in `torch.export` (e.g., torch.split).
                    # This scale factor may be adjusted in the future if additional constraints arise.
                    constraint[dim] = 2 * export_dim

            if not constraints[name]:
                constraints[name] = None

        return cls(
            tensor_inputs=tensor_inputs,
            constant_inputs=constant_inputs,
            constraints=constraints,
        )
