import operator
from collections.abc import Callable
from typing import Any

import tensorrt as trt
import torch
from torch.export._trace import _export as torch_export
from torch.fx import GraphModule, Node
from torch.fx.passes.shape_prop import TensorMetadata
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_tensorrt._enums import dtype

from ._compile import compile
from .arguments_for_export import ArgumentsForExport
from .wrappers import ExportWrapperV2


def convert(
    model: torch.nn.Module,
    arguments: ArgumentsForExport,
    *,
    input_processors: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    output_processors: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    optimal_dims: dict[str, int] | None = None,
    strict: bool = True,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.MATH,
    enable_experimental_decompositions: bool = False,
) -> trt.ICudaEngine:
    with sdpa_kernel(sdp_backends):
        exported_program = torch_export(
            ExportWrapperV2(
                model,
                input_processors=input_processors,
                output_processors=output_processors,
                constant_inputs=arguments.constant_inputs,
            ),
            (),
            arguments.tensor_inputs,
            strict=strict,
            pre_dispatch=pre_dispatch,
        )

        interpreter_result = compile(
            exported_program,
            arg_inputs=(),
            kwarg_inputs=arguments.get_torch_trt_inputs(optimal_sizes=optimal_dims),
            assume_dynamic_shape_support=True,
            enabled_precisions={dtype.f32, dtype.f16},
            enable_experimental_decompositions=enable_experimental_decompositions,
            extra_pre_inline_passes=[
                eliminate_empty_tensors_from_cat_or_stack,
                eliminate_nop_cat_or_stack,
            ],
            extra_post_inline_passes=[
                remove_second_outputs_of_scaled_dot_product_attention,
                remove_assert_scalar,
                replace_operator_sub_by_aten_sub,
            ],
        )
        with trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(interpreter_result.serialized_engine)

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(f"({i}) {name}: {engine.get_tensor_shape(name)}")
        return engine


def remove_second_outputs_of_scaled_dot_product_attention(gm: GraphModule) -> GraphModule:
    for node in gm.graph.nodes:
        if node.target not in (
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
        ):
            continue
        if not (
            len(node.users) == 1
            and (user := list(node.users)[0]).target is operator.getitem
            and len(user.args) == 2
            and user.args[1] == 0
        ):
            print(f"[WARNING] Found a scaled_dot_product_attention node {node} whose second mask output is used")
            continue
        node.target = torch.nn.functional.scaled_dot_product_attention
        user.replace_all_uses_with(node)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def remove_assert_scalar(gm: GraphModule) -> GraphModule:
    nodes_to_remove: list[Node] = []
    for node in gm.graph.nodes:
        if node.target is not torch.ops.aten._assert_scalar.default:
            continue
        nodes_to_remove.append(node)
    for node in nodes_to_remove:
        gm.graph.erase_node(node)
    return gm


def replace_operator_sub_by_aten_sub(gm: GraphModule) -> GraphModule:
    for node in gm.graph.nodes:
        if not (node.target is operator.sub and len(node.args) == 2):
            continue
        lhs, rhs = node.args
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            node.target = torch.ops.aten.sub.Tensor
        elif isinstance(lhs, torch.Tensor) and isinstance(rhs, bool | complex | float | int):
            node.target = torch.ops.aten.sub.Scalar
        elif isinstance(lhs, bool | complex | float | int) and isinstance(rhs, torch.Tensor):
            node.target = torch.ops.aten.sub.Scalar
            node.args = node.args[::-1]
        elif isinstance(lhs, int) and isinstance(rhs, int):
            node.target = torch.ops.aten.sub.int
        elif isinstance(lhs, float) and isinstance(rhs, float):
            node.target = torch.ops.aten.sub.float
        elif isinstance(lhs, float) and isinstance(rhs, complex):
            node.target = torch.ops.aten.sub.float_complex
        elif isinstance(lhs, complex) and isinstance(rhs, float):
            node.target = torch.ops.aten.sub.complex_float
        elif isinstance(lhs, int) and isinstance(rhs, float):
            node.target = torch.ops.aten.sub.int_float
        elif isinstance(lhs, float) and isinstance(rhs, int):
            node.target = torch.ops.aten.sub.float_int
        else:
            node.target = torch.ops.aten.sub.default
    return gm


def eliminate_empty_tensors_from_cat_or_stack(gm: GraphModule) -> GraphModule:
    for node in gm.graph.nodes:
        if not (
            node.target in (torch.ops.aten.cat.default, torch.ops.aten.stack.default)
            and isinstance((tensors := node.args[0]), list | tuple)
            and all(isinstance(tensor, Node) for tensor in tensors)
        ):
            continue
        non_empty_tensors = tuple(
            tensor
            for tensor in tensors
            if isinstance(tensor, Node)
            and isinstance((tensor_meta := tensor.meta.get("tensor_meta", None)), TensorMetadata)
            and tensor_meta.shape.numel() > 0
        )
        node.args = (non_empty_tensors,) + node.args[1:]
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def eliminate_nop_cat_or_stack(gm: GraphModule) -> GraphModule:
    for node in gm.graph.nodes:
        if not (
            node.target in (torch.ops.aten.cat.default, torch.ops.aten.stack.default)
            and isinstance((tensors := node.args[0]), list | tuple)
            and len(tensors) == 1
            and isinstance((the_input := tensors[0]), Node)
        ):
            continue
        node.replace_all_uses_with(the_input)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm
