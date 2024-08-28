import copy
import inspect
from collections.abc import Callable
from functools import cached_property
from types import TracebackType
from typing import Any

import torch
from torch.export import Dim
from torch.utils.hooks import RemovableHandle

from .arguments_for_export import ArgumentsForExport
from .pretty_print import brief_tensor_repr
from .types import BuiltInConstant, DimType


class ShapeHistory(tuple[int | torch.LongTensor, ...]):
    @cached_property
    def dynamic_dims(self) -> dict[int, torch.LongTensor]:
        return {i: v for i, v in enumerate(self) if isinstance(v, torch.LongTensor)}

    @property
    def num_dynamic_dims(self) -> int:
        return len(self.dynamic_dims)

    @cached_property
    def max_shape(self) -> tuple[int, ...]:
        return tuple(x if isinstance(x, int) else x.max().item() for x in self)  # type: ignore

    @cached_property
    def min_shape(self) -> tuple[int, ...]:
        return tuple(x if isinstance(x, int) else x.min().item() for x in self)  # type: ignore

    @cached_property
    def mean_shape(self) -> tuple[int, ...]:
        return tuple(
            x if isinstance(x, int)
            # dynamic dimension example size must be larger than 1
            else max(x.float().mean().long().item(), 2)  # type: ignore
            for x in self
        )


def shape_as_slices(shape: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(0, s) for s in shape)


def get_shape_history(tensors: list[torch.Tensor]) -> ShapeHistory:
    assert len(tensors) > 1
    ndim = tensors[0].ndim
    assert all(ndim == t.ndim for t in tensors), "Expected tensors with all same dimensions"
    shapes = torch.LongTensor([t.shape for t in tensors]).transpose(0, 1)
    is_static_axis: list[bool] = (shapes.min(dim=1).values == shapes.max(dim=1).values).tolist()
    return ShapeHistory(
        shapes[i][0].item() if is_static else shapes[i] for i, is_static in enumerate(is_static_axis)  # type: ignore
    )


class ForwardArgumentCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        preprocess_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        collect_before_forward: bool = True,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.preprocess_inputs = preprocess_inputs
        self.before_forward = collect_before_forward
        self.verbose = verbose
        self._argument_history: dict[str, list[Any]] = {}
        self._count = 0
        self._handle: RemovableHandle | None = None
        self._count_range: range | None = None

    def get(self) -> dict[str, list[Any]]:
        try:
            return self._argument_history
        finally:
            self._argument_history = {}
            self._count = 0

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.model.forward)

    def __len__(self) -> int:
        return self._count

    def record(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if self._count_range is not None and self._count not in self._count_range:
            return False

        kwargs = copy.deepcopy(kwargs)
        if args:
            keys = list(self.signature.parameters.keys())[: len(args)]
            print(
                f"[WARNING] {len(args)} positional arguments passed to the model will be converted as "
                f"keyword arguments with the following keys: {', '.join(keys)}"
            )
            kwargs.update(zip(keys, copy.deepcopy(args)))

        if self.preprocess_inputs:
            kwargs = self.preprocess_inputs(kwargs)

        if self._count == 0:
            self._argument_history = {name: [value] for name, value in kwargs.items()}
        else:
            for name, value in kwargs.items():
                assert name in self._argument_history, f"Previously unseen argument found: {name}"
                self._argument_history[name].append(value)
        self._count += 1
        return True

    def hook(self, _: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], outputs: Any) -> None:
        self.record(args, kwargs)

    def pre_hook(self, _: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.record(args, kwargs)

    def print_readable(self, *, since_iteration: int = 0) -> None:
        with brief_tensor_repr():
            for i in range(since_iteration, self._count):
                print(f"=======================Iteration {i}======================")
                for name, kwargs_history in self._argument_history.items():
                    print(f"{name}: {kwargs_history[i]}")

    def _insert_hook(self) -> None:
        if self._handle:
            self._handle.remove()
        self._handle = (
            self.model.register_forward_pre_hook(self.pre_hook, with_kwargs=True)
            if self.before_forward
            else self.model.register_forward_hook(self.hook, with_kwargs=True)
        )

    def _remove_hook(self) -> None:
        if self._handle:
            self._handle.remove()
        if self.verbose:
            self.print_readable(since_iteration=self._count_range.start if self._count_range is not None else 0)
        self._count_range = None

    def collect(self, max_num_arguments: int) -> "ArgumentCollectorSession":
        self._count_range = range(self._count, self._count + max_num_arguments)
        return ArgumentCollectorSession(self)

    def get_arguments_for_export(
        self,
        start: int | None = None,
        stop: int | None = None,
        dim_names: dict[str, dict[int, str]] | None = None,
        extra_shape_constraints: dict[str, dict[int, DimType]] | None = None,
        device: torch.device | str | None = None,
    ) -> ArgumentsForExport:
        tensor_inputs: dict[str, torch.Tensor] = {}
        constant_inputs: dict[str, BuiltInConstant] = {}
        _constraints: dict[str, dict[int, str]] = {}
        dim_size_history: dict[str, torch.LongTensor] = {}

        def _autogen_name(param_name: str, axis: int) -> str:
            return f"{param_name}_dim_{axis}"

        if device is None:
            device = next(self.model.parameters()).device
        for param, full_argument_history in self._argument_history.items():
            argument_history = full_argument_history[slice(start, stop)]
            tensors = [x for x in argument_history if isinstance(x, torch.Tensor)]
            constants = [x for x in argument_history if isinstance(x, BuiltInConstant)]
            if len(tensors) == len(argument_history):
                shape_history = get_shape_history(tensors)
                last_input = tensors[-1]
                if shape_history.num_dynamic_dims == 0:
                    tensor_inputs[param] = last_input
                    _constraints[param] = {}
                else:
                    argument_constraint: dict[int, str] = {}
                    for axis, size_values in shape_history.dynamic_dims.items():
                        dim_name = _autogen_name(param, axis)
                        argument_constraint[axis] = dim_name
                        dim_size_history[dim_name] = size_values
                    max_size_input = torch.zeros(
                        shape_history.max_shape,
                        device=last_input.device,
                        dtype=last_input.dtype,
                    )
                    max_size_input[shape_as_slices(last_input.shape)] = last_input
                    tensor_inputs[param] = max_size_input[shape_as_slices(shape_history.mean_shape)]
                    _constraints[param] = argument_constraint
            elif len(constants) == len(argument_history):
                if all((value := constants[0]) == x for x in constants):
                    constant_inputs[param] = value
                else:
                    tensor_inputs[param] = torch.tensor(constants[-1], device=device)
                    _constraints[param] = {}
            else:
                raise ValueError(f"Unsupported combination of values provided to {param}: {argument_history}")

        resolved_dims: dict[str, DimType] = {}
        dim_name_map = {}
        if dim_names:
            for param_name, axis_names in dim_names.items():
                for axis, user_defined_name in axis_names.items():
                    dim_name_map[_autogen_name(param_name, axis)] = user_defined_name

        def _create_dim(name: str, min_: int = 0, max_: int = 1 << 32 - 1) -> DimType:
            return Dim(dim_name_map.get(name, name), min=min_, max=max_)

        for dim_name, size_history in dim_size_history.items():
            for existing_dim_name, existing_dim in resolved_dims.items():
                existing_size_history = dim_size_history[existing_dim_name]
                if (existing_size_history == size_history).all():
                    resolved_dims[dim_name] = existing_dim
                    break
                if (diff := size_history - existing_size_history).min() == diff.max():
                    diff_value = diff.min().item()
                    if diff_value > 0:
                        resolved_dims[dim_name] = existing_dim + diff_value
                    else:
                        resolved_dims[dim_name] = (dim := _create_dim(dim_name))
                        resolved_dims[existing_dim_name] = dim + (-diff_value)
                    break
            else:
                resolved_dims[dim_name] = _create_dim(dim_name)
        constraints: dict[str, dict[int, DimType]] = {
            param_name: {i: resolved_dims[dim_name] for i, dim_name in constraint.items()}
            for param_name, constraint in _constraints.items()
        }

        def _update_constraints(
            src: dict[str, dict[int, DimType]],
            other: dict[str, dict[int, DimType]],
        ) -> None:
            for param_name, constraint in other.items():
                if (target_constraint := src.get(param_name)) is None:
                    src.update({param_name: constraint})
                    continue
                target_constraint.update(constraint)

        if extra_shape_constraints is not None:
            _update_constraints(constraints, extra_shape_constraints)

        return ArgumentsForExport(
            tensor_inputs={k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in tensor_inputs.items()},
            constant_inputs=constant_inputs,
            constraints=constraints,
        )


class ArgumentCollectorSession:
    def __init__(self, collector: ForwardArgumentCollector) -> None:
        self.collector = collector

    def __enter__(self) -> None:
        self.collector._insert_hook()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.collector._remove_hook()
