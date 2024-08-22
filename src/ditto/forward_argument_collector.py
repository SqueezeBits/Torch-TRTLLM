import copy
import inspect
from types import TracebackType
from typing import Any

import torch
from torch.export import Dim
from torch.export.dynamic_shapes import _Dim as DimType
from torch.utils.hooks import RemovableHandle
from typing_extensions import Self

from .cache_handler import CacheHandler
from .pretty_print import brief_tensor_repr


class ShapeHistory(tuple[int | torch.LongTensor, ...]):
    @property
    def dynamic_axes(self) -> list[int]:
        return [i for i, v in enumerate(self) if isinstance(v, torch.LongTensor)]

    @property
    def num_dynamic_axes(self) -> int:
        return len(self.dynamic_axes)

    @property
    def dynamic_axis(self) -> int:
        assert len(self.dynamic_axes) == 1
        return self.dynamic_axes[0]

    @property
    def dynamic_axis_history(self) -> torch.LongTensor:
        history = self[self.dynamic_axis]
        assert isinstance(history, torch.LongTensor)
        return history

    @property
    def argmax(self) -> int:
        return self.dynamic_axis_history.argmax(0).item()  # type: ignore

    @property
    def slices(self) -> tuple[slice, ...]:
        return tuple(
            slice(None, None) if isinstance(v, int) else slice(None, v.float().mean().long().item()) for v in self
        )


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
        cache_handler: CacheHandler,
        collect_before_forward: bool = True,
        max_num_arguments: int = 2,
    ) -> None:
        self.model = model
        self.cache_handler = cache_handler
        self.collect_before_forward = collect_before_forward
        self.max_num_arguments = max_num_arguments
        self._argument_history: dict[str, list[Any]] = {}
        self._count: int = 0
        self._handle: RemovableHandle | None = None

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.model.forward)

    def record(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if self._count >= self.max_num_arguments:
            return False
        args, kwargs = self.cache_handler.map_to_tensor(args, kwargs)
        arguments = dict(zip(self.signature.parameters, args))
        arguments.update(kwargs)
        if not self.cache_handler.is_static and isinstance(
            (attention_mask := arguments.pop("attention_mask", None)), torch.Tensor
        ):
            if not isinstance((past_key_values := arguments.get("past_key_values")), torch.Tensor):
                raise ValueError(f"Expected past_key_values to be a tensor but got {past_key_values}")
            if not isinstance((input_ids := arguments.get("input_ids", None)), torch.Tensor):
                raise ValueError(f"Expected input_ids to be a tensor but got {input_ids}")
            input_ids_len = input_ids.shape[-1]
            cache_len = past_key_values.shape[-2]
            assert attention_mask.shape[-1] == input_ids_len + cache_len
            prefilled_attention_mask, generation_attention_mask = torch.split(
                attention_mask, [cache_len, input_ids_len], dim=-1
            )
            arguments["prefilled_attention_mask"] = prefilled_attention_mask
            arguments["generation_attention_mask"] = generation_attention_mask
        if self._count == 0:
            self._argument_history = {name: [copy.deepcopy(value)] for name, value in arguments.items()}
        else:
            for name, value in arguments.items():
                assert name in self._argument_history, f"Previously unseen argument found: {name}"
                self._argument_history[name].append(copy.deepcopy(value))
        self._count += 1
        return True

    def hook(self, _: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], outputs: Any) -> None:
        self.record(args, kwargs)

    def pre_hook(self, _: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.record(args, kwargs)

    def print_readable(self) -> None:
        with brief_tensor_repr():
            for i in range(self._count):
                print(f"=======================Iteration {i}======================")
                for name, kwargs_history in self._argument_history.items():
                    print(f"{name}: {kwargs_history[i]}")

    def get_example_inputs_and_dynamic_shapes(
        self,
        *,
        start: int | None = None,
        stop: int | None = None,
        batch_axis: int = 0,
        batch_dim: DimType | None = None,
        extra_shape_constraints: dict[str, dict[int, DimType] | None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[int, DimType] | None]]:
        assert self._count > 0
        example_inputs: dict[str, Any] = {}
        _constraints: dict[str, dict[int, str] | None] = {}
        dims_and_historys: dict[str, tuple[DimType, torch.LongTensor]] = {}
        for name, full_history in self._argument_history.items():
            history = full_history[slice(start, stop)]
            tensors = [x for x in history if isinstance(x, torch.Tensor)]
            non_tensors = [x for x in history if not isinstance(x, torch.Tensor)]
            if len(tensors) == len(history):
                shape_history = get_shape_history(tensors)
                if shape_history.num_dynamic_axes == 0:
                    example_inputs[name] = tensors[-1]
                    _constraints[name] = None
                elif shape_history.num_dynamic_axes == 1:
                    argmax = shape_history.argmax
                    dynamic_axis = shape_history.dynamic_axis
                    value = tensors[argmax]
                    example_inputs[name] = value[shape_history.slices]
                    dim_name = f"{name}_dim_{dynamic_axis}"
                    dim = Dim(dim_name, min=0, max=1 << 30)
                    _constraints[name] = {dynamic_axis: dim_name}
                    dims_and_historys[dim_name] = (dim, shape_history.dynamic_axis_history)
                else:
                    raise ValueError(f"The tensor values for {name} has more than one dynamic axis: {shape_history}")
            elif len(non_tensors) == len(history):
                if not all((value := non_tensors[0]) == x for x in non_tensors):
                    raise ValueError(f"Expected consistent non-tensor values for {name} but got {non_tensors}")
                example_inputs[name] = value
                _constraints[name] = None
            else:
                raise ValueError(f"Mixed tensor and non-tensor values provided to {name}: {history}")

        resolved_dims: dict[str, DimType] = {}
        for dim_name, (dim, history) in dims_and_historys.items():
            for existing_dim_name, existing_dim in resolved_dims.items():
                existing_dim_history = dims_and_historys[existing_dim_name][1]
                if (existing_dim_history == history).all():
                    resolved_dims[dim_name] = existing_dim
                    break
                elif (diff := history - existing_dim_history).min() == diff.max():
                    diff_value = diff.min().item()
                    if diff_value > 0:
                        resolved_dims[dim_name] = existing_dim + diff_value
                    else:
                        resolved_dims[dim_name] = dim
                        resolved_dims[existing_dim_name] = dim + (-diff_value)
                    break
            else:
                resolved_dims[dim_name] = dim
        constraints: dict[str, dict[int, DimType] | None] = {
            param_name: None
            if constraint is None
            else {i: resolved_dims[dim_name] for i, dim_name in constraint.items()}
            for param_name, constraint in _constraints.items()
        }
        if batch_dim is None:
            batch_dim = Dim("batch", min=0, max=1 << 30)
        batch_constraints: dict[str, dict[int, DimType] | None] = {
            "input_ids": {batch_axis: batch_dim},
            "position_ids": {batch_axis: batch_dim},
            "past_key_values": {batch_axis + 2: batch_dim},
        }
        if self.cache_handler.is_static:
            batch_constraints.update(attention_mask={batch_axis: batch_dim})
        else:
            batch_constraints.update(
                prefilled_attention_mask={batch_axis: batch_dim},
                generation_attention_mask={batch_axis: batch_dim},
            )

        def _update_constraints(
            src: dict[str, dict[int, DimType] | None],
            other: dict[str, dict[int, DimType] | None],
        ) -> None:
            for param_name, constraint in other.items():
                if (target_constraint := src.get(param_name)) is None:
                    src.update({param_name: constraint})
                    continue
                if constraint is None:
                    src.update({param_name: None})
                    continue
                target_constraint.update(constraint)

        _update_constraints(constraints, batch_constraints)
        if extra_shape_constraints is not None:
            _update_constraints(constraints, extra_shape_constraints)
        return example_inputs, constraints

    def __enter__(self) -> Self:
        if self._handle:
            self._handle.remove()
        self._handle = (
            self.model.register_forward_pre_hook(self.pre_hook, with_kwargs=True)
            if self.collect_before_forward
            else self.model.register_forward_hook(self.hook, with_kwargs=True)
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle:
            self._handle.remove()
