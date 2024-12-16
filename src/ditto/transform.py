import gc
from collections.abc import Callable
from itertools import chain
from typing import Any

import torch
import torch.utils._pytree as pytree
from loguru import logger
from torch.fx import GraphModule, Node
from torch.fx.graph import CodeGen, _PyTreeCodeGen
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_POST_LOWERING_PASSES,
    DynamoPassManager,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from .arguments import TRTLLMArgumentHint
from .constants import INPUT_IDS, INPUT_IDS_UNSQUEEZE_DIM, PassName
from .debug import save_for_debug
from .fx.nodes import ATenOp, GetAttr
from .fx.optimize import get_optimization_transform
from .fx.utils import get_tensor_metadata, populate_tensor_metadata
from .pretty_print import detailed_sym_node_str, ignore_symbolic_shapes_warning


def transform(
    graph_module: GraphModule,
    argument_hint: TRTLLMArgumentHint,
    dtype: torch.dtype,
    *,
    skipped_optimizers: list[PassName] | None = None,
    allow_matmul_in_fp16: bool = False,
    allow_activation_in_fp16: bool = True,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
) -> GraphModule:
    post_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [
            fold_constants,
            *(f for f in ATEN_POST_LOWERING_PASSES.passes if f.__name__ not in ("constant_fold", "view_to_reshape")),
        ]
    )

    match_input_ids_dynamic_dims(argument_hint, graph_module)
    logger.debug("Running post-inlining passes")
    with ignore_symbolic_shapes_warning():
        graph_module = post_inline_pass_manager(graph_module)

    save_for_debug("initial_graph_module", graph_module)
    argument_hint.num_attn_layers = count_scaled_dot_product_attention(graph_module)

    custom_pass_manager = DynamoPassManager.build_from_passlist(
        [
            prepare_for_optimization_passes,
            get_optimization_transform(
                argument_hint,
                dtype,
                skipped_optimizers=skipped_optimizers,
                allow_matmul_in_fp16=allow_matmul_in_fp16,
                allow_activation_in_fp16=allow_activation_in_fp16,
            ),
            fold_constants,
            *(extra_passes or []),
        ]
    )
    logger.debug("Running custom passes")
    with ignore_symbolic_shapes_warning():
        return custom_pass_manager(graph_module)


# TODO: fix memory leak from constant folding
@torch.inference_mode()
def fold_constants(graph_module: GraphModule) -> GraphModule:
    graph = graph_module.graph
    foldable_nodes: dict[Node, torch.Tensor] = {}

    def get_qualname() -> str:
        i = 0
        qualname = "folded_constant"
        while hasattr(graph_module, qualname):
            i += 1
            qualname = f"folded_constant_{i}"
        return qualname

    def are_all_input_nodes_fetchable(n: Node) -> bool:
        nonlocal foldable_nodes
        return all(
            (input_node in foldable_nodes or GetAttr.specialize_from(input_node) is not None)
            for input_node in n.all_input_nodes
        )

    def fetch_value(n: Node) -> torch.Tensor:
        nonlocal foldable_nodes
        return get_attr.tensor if (get_attr := GetAttr.specialize_from(n)) else foldable_nodes[n]

    def is_foldable(node: Node) -> bool:
        nonlocal foldable_nodes
        if node in foldable_nodes:
            return True

        if (aten_op := ATenOp.specialize_from(node)) and are_all_input_nodes_fetchable(node):
            flat_inputs, spec = pytree.tree_flatten((node.args, node.kwargs))
            flat_values = tuple(fetch_value(arg) if isinstance(arg, Node) else arg for arg in flat_inputs)
            arg_values, kwarg_values = pytree.tree_unflatten(flat_values, spec)
            foldable_nodes[node] = aten_op.target(*arg_values, **kwarg_values)
            return True

        return False

    nodes_to_replace = [
        node for node in graph.nodes if is_foldable(node) and all(not is_foldable(user) for user in node.users)
    ]

    for node in nodes_to_replace:
        name = get_qualname()
        graph_module.register_buffer(name, foldable_nodes.pop(node))
        with graph.inserting_before(node):
            get_attr = graph.get_attr(name)
            get_attr.meta.update(node.meta)
            node.replace_all_uses_with(get_attr)
            graph.erase_node(node)

    if nodes_to_replace:
        clean_up_graph_after_modifications(graph_module)

    del foldable_nodes, nodes_to_replace
    logger.debug(f"{gc.collect()=}")

    return remove_unused_constants(graph_module)


def remove_unused_constants(graph_module: GraphModule) -> GraphModule:
    referenced_attr_names = {
        target for node in graph_module.graph.nodes if node.op == "get_attr" and isinstance(target := node.target, str)
    }
    unused_attr_names = [
        name for name in chain(graph_module._parameters, graph_module._buffers) if name not in referenced_attr_names
    ]
    for name in unused_attr_names:
        delattr(graph_module, name)

    logger.debug(f"Removed {len(unused_attr_names)} unused attributes {gc.collect()=}")
    return graph_module


def prepare_for_optimization_passes(graph_module: GraphModule) -> GraphModule:
    graph_module.graph.set_codegen(CodeGen())
    for node in graph_module.graph.nodes:
        if isinstance(val := node.meta.get("val"), torch.SymInt):
            continue
        val = node.meta.pop("val", None)
        if not isinstance(node.meta.get("tensor_meta"), TensorMetadata) and isinstance(val, torch.Tensor):
            populate_tensor_metadata(node, val)
    clean_up_graph_after_modifications(graph_module)
    return graph_module


def match_input_ids_dynamic_dims(argument_hint: TRTLLMArgumentHint, graph_module: GraphModule) -> None:
    sync_placeholder_names_with_forward_arg_names(graph_module)
    if num_tokens_sym_int := get_input_ids_dynamic_dim(graph_module):
        argument_hint.num_tokens.sym_int = num_tokens_sym_int
        with detailed_sym_node_str():
            logger.debug(f"Matched {repr(argument_hint.num_tokens)} to {num_tokens_sym_int}")
    else:
        logger.warning(f"Failed to match dynamic dimension of {INPUT_IDS}")


def sync_placeholder_names_with_forward_arg_names(graph_module: GraphModule) -> None:
    def _impl(obj: Any) -> None:
        if isinstance(obj, tuple | list):
            for x in obj:
                _impl(x)
            return
        if isinstance(obj, dict):
            for name, value in obj.items():
                if isinstance(name, str) and isinstance(value, Node):
                    logger.debug(f"Renaming placholder '{value}' as forward argument name '{name}'")
                    value.name = name
                    value.target = name
                    continue
                _impl(value)
            return

    if isinstance(codegen := graph_module.graph._codegen, _PyTreeCodeGen):
        inputs = pytree.tree_unflatten(
            graph_module.graph.find_nodes(op="placeholder"),
            codegen.pytree_info.in_spec,
        )
        _impl(inputs)


def get_input_ids_dynamic_dim(graph_module: GraphModule) -> torch.SymInt | None:
    if (
        (placeholders := {p.name: p for p in graph_module.graph.find_nodes(op="placeholder")})
        and INPUT_IDS in placeholders
        and (meta := get_tensor_metadata(placeholders[INPUT_IDS]))
        and isinstance(sym_int := meta.shape[1 - INPUT_IDS_UNSQUEEZE_DIM], torch.SymInt)
    ):
        return sym_int
    return None


def count_scaled_dot_product_attention(graph_module: GraphModule) -> int:
    return len(
        [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target is torch._C._nn.scaled_dot_product_attention
        ]
    )
