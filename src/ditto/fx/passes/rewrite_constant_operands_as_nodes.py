import keyword
import re
from typing import Literal

import torch
from torch.fx import GraphModule, Node

from ...types import Number
from ..nodes import Binary
from ..utils import get_tensor_metadata, populate_tensor_metadata
from .node_wise_pass import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses


class RewriteConstantOperandsAsNodes(NodewiseOptimizationPass):
    """Rewrite constant operands of binary nodes as nodes.

    This pass is required for avoiding failure in converting a constant operand into numpy array by the interpreter
    when the constant's data type is not supported by numpy (e.g. bfloat16).
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (binary := Binary.specialize_from(node))
            and len(node.args) >= 2
            and (graph_module := (graph := node.graph).owning_module)
        ):
            return {}

        constant_idx: Literal[0, 1]
        name: str
        param: torch.nn.Parameter
        if isinstance(binary.this, Number) and isinstance(binary.other, Node):
            constant_idx = 0
            name, param = get_or_register_parameter(graph_module, node=binary.other, constant=binary.this)
        elif isinstance(binary.other, Number) and isinstance(binary.this, Node):
            constant_idx = 1
            name, param = get_or_register_parameter(graph_module, constant=binary.other, node=binary.this)
        else:
            return {}

        with graph.inserting_before(node):
            constant = graph.get_attr(name)
            populate_tensor_metadata(constant, param)
            new_args = (constant, *node.args[1:]) if constant_idx == 0 else (node.args[0], constant, *node.args[2:])
            rewritten_binary = graph.call_function(binary.target, new_args, node.kwargs)
            if node.stack_trace:
                rewritten_binary.stack_trace = f"{node.stack_trace}, pass: rewritten by {__name__}"
            if t := get_tensor_metadata(node):
                populate_tensor_metadata(rewritten_binary, t)
        return {node: ReplaceAllUses(by=rewritten_binary)}


def get_or_register_parameter(
    graph_module: GraphModule,
    *,
    node: Node,
    constant: Number,
) -> tuple[str, torch.nn.Parameter]:
    name = make_as_identifier(f"literal_{constant}")
    try:
        param = graph_module.get_parameter(name)
    except AttributeError:
        dtype: torch.dtype | None = None
        if meta := get_tensor_metadata(node):
            dtype = meta.dtype
        param = torch.nn.Parameter(torch.tensor(constant, dtype=dtype), requires_grad=False)
        graph_module.register_parameter(name, param)
    return name, param


def make_as_identifier(s: str) -> str:
    # Remove invalid characters and replace them with underscores
    s = re.sub(r"\W|^(?=\d)", "_", s)

    # If the result is a Python keyword, add a suffix to make it a valid identifier
    if keyword.iskeyword(s):
        s += "_id"

    return s
