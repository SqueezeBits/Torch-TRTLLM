from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Generator
from typing import Any

import torch
import torch.utils._pytree as pytree
from loguru import logger
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import log as symbolic_shape_logger
from torch.fx.graph import CodeGen, _PyTreeCodeGen
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo.lowering import get_decompositions
from torch_tensorrt.dynamo.lowering.passes import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
    DynamoPassManager,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from .arguments import TRTLLMArgumentHint
from .constants import INPUT_IDS_UNSQUEEZE_DIM, PassName
from .fx.optimize import get_optimization_transform
from .fx.utils import get_tensor_metadata, populate_tensor_metadata
from .pretty_print import detailed_sym_node_str


def inline(
    exported_program: ExportedProgram,
    argument_hint: TRTLLMArgumentHint,
    *,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    skipped_optimizers: list[PassName] | None = None,
    allow_matmul_in_fp16: bool = False,
    allow_activation_in_fp16: bool = True,
    extra_passes: list[Callable[[GraphModule], GraphModule]] | None = None,
) -> GraphModule:
    pretrained_config = exported_program.graph_module.meta.get("pretrained_config", None)
    pre_inline_pass_manager = DynamoPassManager.build_from_passlist(ATEN_PRE_LOWERING_PASSES.passes)
    post_inline_pass_manager = DynamoPassManager.build_from_passlist(
        [f for f in ATEN_POST_LOWERING_PASSES.passes if f.__name__ != "view_to_reshape"]
    )

    graph_module: GraphModule
    with ignore_symbolic_shapes_warning():
        _ = pre_inline_pass_manager(exported_program.graph_module)
        exported_program = exported_program.run_decompositions(get_decompositions(enable_experimental_decompositions))
        graph_module = exported_program.module()  # type: ignore[assignment]
        if num_tokens_sym_int := get_input_ids_dynamic_dim(graph_module):
            argument_hint.num_tokens.sym_int = num_tokens_sym_int
            with detailed_sym_node_str():
                logger.debug(f"Matched {repr(argument_hint.num_tokens)} to {num_tokens_sym_int}")
        graph_module = post_inline_pass_manager(graph_module)

    argument_hint.num_attn_layers = count_scaled_dot_product_attention(graph_module)

    custom_pass_manager = DynamoPassManager.build_from_passlist(
        [
            prepare_for_optimization_passes,
            get_optimization_transform(
                argument_hint,
                skipped_optimizers=skipped_optimizers,
                allow_matmul_in_fp16=allow_matmul_in_fp16,
                allow_activation_in_fp16=allow_activation_in_fp16,
            ),
            *(extra_passes or []),
        ]
    )

    with ignore_symbolic_shapes_warning():
        graph_module = custom_pass_manager(graph_module)

    graph_module.meta["pretrained_config"] = pretrained_config
    assert isinstance(graph_module, GraphModule)
    graph_module._forward_pre_hooks.clear()
    return graph_module


@contextlib.contextmanager
def ignore_symbolic_shapes_warning() -> Generator[None, None, None]:
    log_level = symbolic_shape_logger.level
    symbolic_shape_logger.setLevel(logging.ERROR)
    try:
        yield None
    finally:
        symbolic_shape_logger.setLevel(log_level)


def prepare_for_optimization_passes(graph_module: GraphModule) -> GraphModule:
    sync_placeholder_names_with_forward_arg_names(graph_module)
    graph_module.graph._codegen = CodeGen()
    for node in graph_module.graph.nodes:
        if isinstance(val := node.meta.get("val"), torch.SymInt):
            continue
        val = node.meta.pop("val", None)
        if not isinstance(node.meta.get("tensor_meta"), TensorMetadata) and isinstance(val, torch.Tensor):
            populate_tensor_metadata(node, val)
    clean_up_graph_after_modifications(graph_module)
    return graph_module


def sync_placeholder_names_with_forward_arg_names(graph_module: GraphModule) -> None:
    def _impl(obj: Any) -> None:
        if isinstance(obj, tuple | list):
            for x in obj:
                _impl(x)
            return
        if isinstance(obj, dict):
            for name, value in obj.items():
                if isinstance(name, str) and isinstance(value, Node):
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


def count_scaled_dot_product_attention(graph_module: GraphModule) -> int:
    return len(
        [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target is torch._C._nn.scaled_dot_product_attention
        ]
    )


def get_input_ids_dynamic_dim(graph_module: GraphModule) -> torch.SymInt | None:
    if (
        (placeholders := graph_module.graph.find_nodes(op="placeholder"))
        and (meta := get_tensor_metadata(placeholders[0]))
        and isinstance(sym_int := meta.shape[1 - INPUT_IDS_UNSQUEEZE_DIM], torch.SymInt)
    ):
        return sym_int
    logger.warning("Failed to extract input_ids dynamic dimension")
    return None
