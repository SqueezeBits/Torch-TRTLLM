import copy
import inspect
from collections.abc import Callable
from functools import cached_property
from types import TracebackType
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle

from .arguments_for_export import ArgumentsForExport
from .dynamic_dim import DynamicDimension, DynamicDimensionType
from .pretty_print import brief_tensor_repr
from .types import BuiltInConstant, DimType, StrictlyTyped


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


class TensorArgumentHistories(dict[str, list[torch.Tensor]]):
    @property
    def shape_histories(self) -> dict[str, ShapeHistory]:
        return {name: get_shape_history(self[name]) for name in self}


class ArgumentHistory(StrictlyTyped):
    dynamic: TensorArgumentHistories
    static: TensorArgumentHistories
    constants: dict[str, BuiltInConstant]
    all_keys: tuple[str, ...]

    def resolve_dynamic_dims(
        self,
        predefined_dims: dict[str, dict[int, DynamicDimensionType]] | None = None,
        default_min: int = 0,
        default_max: int = 1 << 15 - 1,
        default_opt: int | None = None,
        default_example_for_export: int | None = None,
    ) -> tuple[dict[str, dict[int, DimType] | None], dict[str, dict[int, int]], dict[str, torch.Tensor]]:
        def _autogen_name(param_name: str, axis: int) -> str:
            return f"{param_name}_dim_{axis}"

        _constraints: dict[str, dict[int, str]] = {}
        dim_size_history: dict[str, torch.LongTensor] = {}
        for name, shape_history in self.dynamic.shape_histories.items():
            argument_constraint: dict[int, str] = {}
            for axis, size_values in shape_history.dynamic_dims.items():
                dim_name = _autogen_name(name, axis)
                argument_constraint[axis] = dim_name
                dim_size_history[dim_name] = size_values
                _constraints[name] = argument_constraint

        resolved_dims: dict[str, DynamicDimensionType] = {}
        if predefined_dims:
            for param_name, axis_and_dims in predefined_dims.items():
                for axis, dynamic_dim in axis_and_dims.items():
                    dim_name = _autogen_name(param_name, axis)
                    resolved_dims[dim_name] = dynamic_dim

        for dim_name, size_history in dim_size_history.items():
            if dim_name in resolved_dims:
                continue
            for existing_dim_name, existing_dim in resolved_dims.items():
                if existing_dim_name not in dim_size_history:
                    print(f"[WARNING] The predefined dynamic dimension {existing_dim.name} is actually static")
                    continue
                existing_size_history = dim_size_history[existing_dim_name]
                if (existing_size_history == size_history).all():
                    resolved_dims[dim_name] = existing_dim
                    break
                if (diff := size_history - existing_size_history).min() == diff.max() and isinstance(
                    (diff_value := diff.min().item()), int
                ):
                    resolved_dims[dim_name] = existing_dim + diff_value
                    break
            else:
                readable_dim_name = dim_name.replace("_dim_", "'s dimension ")
                print(f"[WARNING] No predefined dynamic dimension given for {readable_dim_name}.")
                resolved_dims[dim_name] = DynamicDimension(
                    name=dim_name,
                    min=default_min,
                    opt=default_opt or int(size_history.float().mean().item()),
                    max=default_max,
                    example_for_export=default_example_for_export,
                )
        constraints: dict[str, dict[int, DimType] | None] = {
            param: {i: resolved_dims[dim_name].export_dim for i, dim_name in constraint.items()}
            for param, constraint in _constraints.items()
        }
        example_sizes = {
            param: {i: resolved_dims[dim_name].example for i, dim_name in constraint.items()}
            for param, constraint in _constraints.items()
        }
        optimal_sizes = {
            param: {i: resolved_dims[dim_name].opt for i, dim_name in constraint.items()}
            for param, constraint in _constraints.items()
        }
        input_tensors: dict[str, torch.Tensor] = {}
        for param, tensors in self.dynamic.items():
            example_size = example_sizes.get(param, {})
            last_tensor = tensors[-1]
            opt_shape = tuple(example_size.get(axis, last_tensor.shape[axis]) for axis in range(last_tensor.ndim))
            opt_tensor = torch.zeros(*opt_shape, dtype=last_tensor.dtype, device=last_tensor.device)
            slices = shape_as_slices(tuple(min(x, y) for x, y in zip(opt_shape, last_tensor.shape)))
            opt_tensor[slices] = last_tensor[slices]
            input_tensors[param] = opt_tensor
        for param, tensors in self.static.items():
            input_tensors[param] = tensors[-1]
            constraints[param] = None

        return constraints, optimal_sizes, input_tensors


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
        self._num_existing_arguments: int = 0

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
        if not self._argument_history:
            return 0
        return len([*self._argument_history.values()][0])

    def record(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if self._count_range is not None and self._count not in self._count_range:
            self._count += 1
            return False

        kwargs = copy.deepcopy(kwargs)
        if args:
            keys = list(self.signature.parameters.keys())[: len(args)]
            print(
                f"[WARNING] {len(args)} positional arguments passed to the model will be converted as "
                f"keyword arguments with the following keys: {', '.join(keys)}"
            )
            kwargs = {**dict(zip(keys, copy.deepcopy(args))), **kwargs}

        if self.preprocess_inputs:
            kwargs = self.preprocess_inputs(kwargs)

        if len(self) == 0:
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

    def print_readable(self) -> None:
        with brief_tensor_repr():
            for idx, iteration in enumerate(self._count_range or range(self._count)):
                print(f"=======================Iteration {iteration}======================")
                for name, kwargs_history in self._argument_history.items():
                    print(f"{name}: {kwargs_history[self._num_existing_arguments + idx]}")

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
            self.print_readable()
        self._count_range = None
        self._num_existing_arguments = len(self)

    def collect(self, max_num_arguments: int, skip_first_n: int = 0) -> "ArgumentCollectorSession":
        self._count_range = range(self._count + skip_first_n, self._count + skip_first_n + max_num_arguments)
        return ArgumentCollectorSession(self)

    def get_argument_history(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> ArgumentHistory:
        dynamic_argument_histories = TensorArgumentHistories()
        static_argument_histories = TensorArgumentHistories()
        constant_arguments: dict[str, BuiltInConstant] = {}
        for name, full_argument_history in self._argument_history.items():
            argument_history = full_argument_history[slice(start, stop, step)]
            tensors = [x for x in argument_history if isinstance(x, torch.Tensor)]
            constants = [x for x in argument_history if isinstance(x, BuiltInConstant)]
            if len(tensors) == len(argument_history):
                shape_history = get_shape_history(tensors)
                if shape_history.num_dynamic_dims == 0:
                    static_argument_histories[name] = tensors
                else:
                    dynamic_argument_histories[name] = tensors
            elif len(constants) == len(argument_history):
                if all((value := constants[0]) == x for x in constants):
                    constant_arguments[name] = value
                else:
                    raise ValueError(
                        f"Expected constant input {name} to have consistent values, but got {argument_history}"
                    )
                # else:
                #     static_argument_history[name] = [torch.tensor(constant, device=device) for constant in constants]
            else:
                with brief_tensor_repr():
                    raise ValueError(f"Unsupported combination of values provided to {name}: {argument_history}")
        return ArgumentHistory(
            dynamic=dynamic_argument_histories,
            static=static_argument_histories,
            constants=constant_arguments,
            all_keys=tuple(self._argument_history.keys()),
        )

    def get_arguments_for_export(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        *,
        dynamic_dims: dict[str, dict[int, DynamicDimensionType]] | None = None,
        default_min: int = 0,
        default_max: int = 1 << 15 - 1,
        default_opt: int | None = None,
        default_example_for_export: int | None = None,
    ) -> ArgumentsForExport:
        argument_history = self.get_argument_history(start, stop, step)
        constraints, optimial_sizes, tensor_inputs = argument_history.resolve_dynamic_dims(
            predefined_dims=dynamic_dims,
            default_min=default_min,
            default_max=default_max,
            default_opt=default_opt,
            default_example_for_export=default_example_for_export,
        )

        return ArgumentsForExport(
            tensor_inputs={k: v.contiguous() for k, v in tensor_inputs.items()},
            constant_inputs=argument_history.constants,
            constraints=constraints,
            optimal_sizes=optimial_sizes,
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
