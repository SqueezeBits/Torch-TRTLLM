from typing import TypedDict

import torch
from pydantic import Field
from torch_tensorrt._Input import Input
from typing_extensions import Self

from .constants import DEFAULT_DEVICE
from .dynamic_dim import DynamicDimension, DynamicDimensionType
from .pretty_print import brief_tensor_repr
from .types import BuiltInConstant, DimType, StrictlyTyped


class InputHint(TypedDict):
    shape: tuple[int | DynamicDimensionType, ...]
    dtype: torch.dtype
    device: torch.device | str


class ArgumentsForExport(StrictlyTyped):
    tensor_inputs: dict[str, torch.Tensor]
    constant_inputs: dict[str, BuiltInConstant] = Field(default_factory=dict)
    constraints: dict[str, dict[int, DimType] | None] = Field(default_factory=dict)
    optimal_sizes: dict[str, dict[int, int]] = Field(default_factory=dict)

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
                if constraint is None:
                    print(f"{name}: None")
                    continue
                dim_reprs = {axis: dim_repr(dim) for axis, dim in constraint.items()}
                print(f"{name}: {dim_reprs}")
            print("========================================================")

    @classmethod
    def get_trtllm_inputs(
        cls,
        device: torch.device | str = DEFAULT_DEVICE,
        **other_flags: BuiltInConstant,
    ) -> Self:
        batch_size = DynamicDimension(
            name="batch_size",
            min=1,
            opt=128,
            max=256,
        )
        s = DynamicDimension(name="s", min=0, opt=32, max=1024)
        input_size = 8 * s
        block_size = DynamicDimension(
            name="block_size",
            min=1,
            opt=32,
            max=64,
        )
        cache_indirection_size = DynamicDimension(
            name="cache_indirection_size",
            min=1,
            opt=2048,
            max=4096,
        )
        hints = {
            "input_ids": InputHint(shape=(input_size,), dtype=torch.int32, device=device),
            "position_ids": InputHint(shape=(input_size,), dtype=torch.int32, device=device),
            "last_token_ids": InputHint(shape=(batch_size,), dtype=torch.int32, device=device),
            "kv_cache_block_offsets": InputHint(shape=(batch_size, 2, block_size), dtype=torch.int32, device=device),
            "host_kv_cache_block_offsets": InputHint(
                shape=(batch_size, 2, block_size), dtype=torch.int32, device=device
            ),
            "host_kv_cache_pool_pointers": InputHint(shape=(2,), dtype=torch.int64, device=device),
            "sequence_length": InputHint(shape=(batch_size,), dtype=torch.int32, device=device),
            "host_request_types": InputHint(shape=(batch_size,), dtype=torch.int32, device=device),
            "host_past_key_value_lengths": InputHint(shape=(batch_size,), dtype=torch.int32, device=device),
            "context_lengths": InputHint(shape=(batch_size,), dtype=torch.int32, device=device),
            "host_runtime_perf_knobs": InputHint(shape=(16,), dtype=torch.int64, device=device),
            "host_context_lengths": InputHint(shape=(batch_size,), dtype=torch.int32, device=device),
            "host_max_attention_window_sizes": InputHint(shape=(32,), dtype=torch.int32, device=device),
            "host_sink_token_length": InputHint(shape=(1,), dtype=torch.int32, device=device),
            "cache_indirection": InputHint(
                shape=(batch_size, 1, cache_indirection_size), dtype=torch.int32, device=device
            ),
        }
        return cls.from_hints(**hints, **other_flags)

    @classmethod
    def from_hints(cls, **input_hints: InputHint | BuiltInConstant) -> Self:
        tensor_inputs: dict[str, torch.Tensor] = {}
        constant_inputs: dict[str, BuiltInConstant] = {}
        constraints: dict[str, dict[int, DimType] | None] = {}
        optimal_sizes: dict[str, dict[int, int]] = {}

        for name, hint in input_hints.items():
            if isinstance(hint, BuiltInConstant):
                constant_inputs[name] = hint
                continue
            shape = tuple(s if isinstance(s, int) else s.example for s in hint["shape"])
            tensor_inputs[name] = torch.zeros(*shape, dtype=hint["dtype"], device=hint["device"])

            for dim, s in enumerate(hint["shape"]):
                if name not in constraints:
                    constraints[name] = {}
                if not isinstance(s, DynamicDimensionType):
                    continue
                assert (constraint := constraints[name]) is not None
                constraint[dim] = s.export_dim

            if not constraints[name]:
                constraints[name] = None

            optimal_sizes[name] = {
                dim: s.opt for dim, s in enumerate(hint["shape"]) if isinstance(s, DynamicDimensionType)
            }

        return cls(
            tensor_inputs=tensor_inputs,
            constant_inputs=constant_inputs,
            constraints=constraints,
            optimal_sizes=optimal_sizes,
        )

    @property
    def torch_trt_inputs(self) -> dict[str, Input]:
        trt_inputs: dict[str, Input] = {}
        for name, tensor in self.tensor_inputs.items():
            # pylint: disable-next=unsupported-membership-test
            if name not in self.constraints:
                trt_input = Input.from_tensor(tensor)
                trt_input.name = name
                trt_inputs[name] = trt_input
                continue
            constraint = self.constraints.get(name, {})
            opt_sizes = self.optimal_sizes.get(name, {})
            min_shape = tuple(
                (getattr(constraint[dim], "min", size) if constraint and dim in constraint else size)
                for dim, size in enumerate(tensor.shape)
            )
            max_shape = tuple(
                (getattr(constraint[dim], "max", size) if constraint and dim in constraint else size)
                for dim, size in enumerate(tensor.shape)
            )
            opt_shape = tuple(opt_sizes.get(dim, size) for dim, size in enumerate(tensor.shape))
            format_ = (
                torch.contiguous_format
                if tensor.is_contiguous(memory_format=torch.contiguous_format)
                else torch.channels_last
            )
            trt_inputs[name] = Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=tensor.dtype,
                format=format_,
                torch_tensor=tensor,
                name=name,
            )
        return trt_inputs
