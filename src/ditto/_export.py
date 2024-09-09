import operator

# import tensorrt as trt
import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch.export._trace import _export as torch_export
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import _PyTreeCodeGen
from torch.nn.attention import SDPBackend, sdpa_kernel

# from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input

# from .compile import convert_to_trt_engine
from .arguments_for_export import ArgumentsForExport
from .cache_handler import CacheHandler
from .wrappers import PostExportWrapper, PreExportWrapper


def export(
    cache_handler: CacheHandler,
    model: torch.nn.Module,
    arguments: ArgumentsForExport,
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.MATH,
    # enable_experimental_decompositions: bool = False,
) -> PostExportWrapper:
    with sdpa_kernel(sdp_backends):
        exported_program = torch_export(
            PreExportWrapper(model, cache_handler=cache_handler, constant_inputs=arguments.constant_inputs),
            (),
            arguments.tensor_inputs,
            dynamic_shapes={"kwargs": arguments.constraints},
            strict=strict,
            pre_dispatch=pre_dispatch,
        )

        # interpreter_result = convert_to_trt_engine(
        #     exported_program,
        #     arg_inputs=(),
        #     kwarg_inputs=arguments.tensor_inputs,
        #     assume_dynamic_shape_support=True,
        #     enabled_precisions={dtype.f32, dtype.f16},
        #     enable_experimental_decompositions=enable_experimental_decompositions,
        #     extra_post_inline_passes=[
        #         remove_second_outputs_of_scaled_dot_product_attention,
        #         remove_assert_scalar,
        #         replace_operator_sub_by_aten_sub,
        #         fuse_attention_mask_inputs,
        #     ],
        #     pre_interpretation_input_modifier=modify_inputs_for_trt_interpreter,
        # )
        # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # with trt.Runtime(TRT_LOGGER) as runtime:
        #     engine = runtime.deserialize_cuda_engine(interpreter_result.serialized_engine)

        # with open("mistral.bin", "wb") as f:
        #     f.write(interpreter_result.serialized_engine)
        # import pdb; pdb.set_trace()

    graph_module = exported_program.module()
    assert isinstance(graph_module, GraphModule)
    graph_module = fuse_attention_mask_inputs(graph_module)
    return PostExportWrapper(graph_module, cache_handler=cache_handler)


def fuse_attention_mask_inputs(graph_module: GraphModule) -> GraphModule:
    graph = graph_module.graph
    placeholders = get_placeholders(graph)
    if not (
        (prefilled_attention_mask := placeholders.get("prefilled_attention_mask")) is not None
        and (generation_attention_mask := placeholders.get("generation_attention_mask")) is not None
        and len(prefilled_attention_mask.users) == 1
        and len(generation_attention_mask.users) == 1
        and (user := [*prefilled_attention_mask.users][0]) in generation_attention_mask.users
        and isinstance((target := user.target), OpOverload)
        and (target._namespace, target._opname, target._overloadname) == ("aten", "cat", "default")
    ):
        return graph_module

    with graph.inserting_after(user):
        attention_mask = graph.placeholder("attention_mask")
        attention_mask.meta = {"val": user.meta["val"], "tensor_meta": user.meta["tensor_meta"]}
    user.replace_all_uses_with(attention_mask)
    graph.erase_node(user)
    graph.erase_node(prefilled_attention_mask)
    graph.erase_node(generation_attention_mask)

    def _modify_names(names: list[str]) -> None:
        names.remove("prefilled_attention_mask")
        names.remove("generation_attention_mask")
        names.append("attention_mask")

    if isinstance((forward_arg_names := graph_module.meta.get("forward_arg_names", None)), list):
        _modify_names(forward_arg_names)

    if isinstance((codegen := graph._codegen), _PyTreeCodeGen):
        target_index: int | None = None
        new_child_spec: pytree.TreeSpec | None = None
        for i, child_spec in enumerate(codegen.pytree_info.in_spec.children_specs):
            if (
                child_spec.type is dict
                and isinstance((context := child_spec.context), list)
                and all(isinstance(x, str) for x in context)
            ):
                target_index = i
                new_context = [*context]
                _modify_names(new_context)
                _, new_child_spec = pytree.tree_flatten_with_path(dict(zip(new_context, new_context)))
                break
        if target_index is not None and new_child_spec is not None:
            codegen.pytree_info.in_spec.children_specs[target_index] = new_child_spec
            graph_module._forward_pre_hooks.clear()

    graph.lint()
    graph.eliminate_dead_code()
    graph_module.recompile()
    return graph_module


def get_placeholders(graph: Graph) -> dict[str, Node]:
    return {node.name: node for node in graph.nodes if node.op == "placeholder"}


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


def modify_inputs_for_trt_interpreter(
    flattened_input_list: list[Input],
    arg_inputs: list[Input],
    kwarg_inputs: dict[str, Input],
) -> tuple[list[Input], list[Input], dict[str, Input]]:
    if (
        (prefilled_attention_mask_spec := kwarg_inputs.pop("prefilled_attention_mask", None))
        and (generation_attention_mask_spec := kwarg_inputs.pop("generation_attention_mask", None))
        and isinstance((prefilled_shape := prefilled_attention_mask_spec.shape), tuple)
        and isinstance((generation_shape := generation_attention_mask_spec.shape), tuple)
    ):
        batch_size = prefilled_shape[0]
        seq_len = prefilled_shape[1] + generation_shape[1]
        attention_mask = Input(
            shape=(batch_size, seq_len),
            dtype=prefilled_attention_mask_spec.dtype,
            format=prefilled_attention_mask_spec.format,
            tensor_domain=prefilled_attention_mask_spec.tensor_domain,
        )
        kwarg_inputs["attention_mask"] = attention_mask
        flattened_input_list = [
            x for x in flattened_input_list if x not in (prefilled_attention_mask_spec, generation_attention_mask_spec)
        ]
        flattened_input_list.append(attention_mask)

    def mark_as_dynamic(
        trt_input: Input,
        *,
        min_shape: tuple[int, ...],
        opt_shape: tuple[int, ...],
        max_shape: tuple[int, ...],
    ) -> None:
        trt_input.shape = {
            "min_shape": min_shape,
            "opt_shape": opt_shape,
            "max_shape": max_shape,
        }
        trt_input.shape_mode = Input._ShapeMode.DYNAMIC

    min_batch = 1
    max_batch = 2
    min_q = 1
    max_q = 512
    min_kv = 0
    max_kv = 512
    if input_ids := kwarg_inputs.get("input_ids", None):
        mark_as_dynamic(
            input_ids,
            min_shape=(min_batch, min_q),
            opt_shape=(1, 1),
            max_shape=(max_batch, max_q),
        )
    if position_ids := kwarg_inputs.get("position_ids", None):
        mark_as_dynamic(
            position_ids,
            min_shape=(min_batch, min_q),
            opt_shape=(1, 1),
            max_shape=(max_batch, max_q),
        )
    if cache_position := kwarg_inputs.get("cache_position", None):
        mark_as_dynamic(
            cache_position,
            min_shape=(min_q,),
            opt_shape=(1,),
            max_shape=(max_q,),
        )
    if past_key_values := kwarg_inputs.get("past_key_values", None):
        mark_as_dynamic(
            past_key_values,
            min_shape=(2, 32, min_batch, 32, min_kv, 128),
            opt_shape=(2, 32, 1, 32, 256, 128),
            max_shape=(2, 32, max_batch, 32, max_kv, 128),
            # min_shape=(2, 32, 1, 8, 0, 128),
            # opt_shape=(2, 32, 1, 8, 256, 128),
            # max_shape=(2, 32, 1, 8, 4096, 128),
        )
    if attention_mask := kwarg_inputs.get("attention_mask", None):
        mark_as_dynamic(
            attention_mask,
            min_shape=(min_batch, min_q + min_kv),
            opt_shape=(1, 257),
            max_shape=(max_batch, max_q + max_kv),
        )
    return flattened_input_list, arg_inputs, kwarg_inputs
