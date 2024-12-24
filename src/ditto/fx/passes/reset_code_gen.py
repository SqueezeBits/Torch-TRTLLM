from typing import Any

import torch.utils._pytree as pytree
from loguru import logger
from torch.fx import GraphModule, Node
from torch.fx.graph import CodeGen, _PyTreeCodeGen

from .infra import GraphOptimizationPass, PassResult


class ResetCodeGen(GraphOptimizationPass):
    """Reset the codegen of the graph to forget all custom input/output structures."""

    def call(self, graph_module: GraphModule) -> PassResult:
        sync_placeholder_names_with_forward_arg_names(graph_module)
        graph_module.graph.set_codegen(CodeGen())
        return PassResult(graph_module=graph_module, modified=True)


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
