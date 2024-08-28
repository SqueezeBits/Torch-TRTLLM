import torch
from pydantic import BaseModel, Field

from .pretty_print import brief_tensor_repr
from .types import BuiltInConstant, DimType


class ArgumentsForExport(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    tensor_inputs: dict[str, torch.Tensor]
    constraints: dict[str, dict[int, DimType]] = Field(default_factory=dict)
    constant_inputs: dict[str, BuiltInConstant] = Field(default_factory=dict)

    def print_readable(self) -> None:
        def dim_repr(dim: DimType) -> str:
            min_ = getattr(dim, "min", "?")
            max_ = getattr(dim, "max", "?")
            return f"{dim.__name__}(min={min_}, max={max_})"

        with brief_tensor_repr():
            print("=============Tensor inputs for torch.export=============")
            for name, tensor in self.tensor_inputs.items():
                print(f"{name}: {tensor}")
            print("============Constant inputs for torch.export============")
            for name, constant in self.constant_inputs.items():
                print(f"{name}: {constant}")
            print("======================Constraints=======================")
            for name, constraint in self.constraints.items():
                dim_reprs = {axis: dim_repr(dim) for axis, dim in constraint.items()}
                print(f"{name}: {dim_reprs}")
            print("========================================================")
