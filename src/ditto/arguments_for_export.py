import torch
from pydantic import BaseModel, Field
from torch_tensorrt._Input import Input

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

    def get_torch_trt_inputs(self, optimal_sizes: dict[str, int] | None = None) -> dict[str, Input]:
        trt_inputs: dict[str, Input] = {}
        opt_sizes = optimal_sizes or {}
        for name, tensor in self.tensor_inputs.items():
            if name not in self.constraints:
                trt_input = Input.from_tensor(tensor)
                trt_input.name = name
                trt_inputs[name] = trt_input
                continue
            constraint = self.constraints[name]
            min_shape = tuple(
                (getattr(constraint[dim], "min", size) if dim in constraint else size)
                for dim, size in enumerate(tensor.shape)
            )
            max_shape = tuple(
                (getattr(constraint[dim], "max", size) if dim in constraint else size)
                for dim, size in enumerate(tensor.shape)
            )
            opt_shape = tuple(
                (
                    opt_sizes[dim_name]
                    if (dim in constraint and (dim_name := getattr(constraint[dim], "__name__", "")) in opt_sizes)
                    else size
                )
                for dim, size in enumerate(tensor.shape)
            )
            format = (
                torch.contiguous_format
                if tensor.is_contiguous(memory_format=torch.contiguous_format)
                else torch.channels_last
            )
            trt_inputs[name] = Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=tensor.dtype,
                format=format,
                torch_tensor=tensor,
                name=name,
            )
        return trt_inputs
