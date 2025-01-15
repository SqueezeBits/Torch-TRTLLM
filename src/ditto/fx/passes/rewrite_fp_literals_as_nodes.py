import keyword
import re

import torch
from torch.fx import Node

from ..nodes import Binary, GetAttr
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, inject_stack_trace_from


class RewriteFloatingPointLiteralsAsNodes(NodewiseOptimizationPass):
    """Rewrite floating point constant literals of binary nodes as nodes.

    This pass is required for avoiding failure in converting a constant operand into numpy array by the interpreter
    when the constant's data type is not supported by numpy (e.g. bfloat16).
    """

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not ((binary := Binary.specialize_from(node)) and len(node.args) >= 2):
            return {}

        graph = node.graph
        constant_key: str
        name: str
        buffer: torch.Tensor
        if isinstance(binary.this, float) and isinstance(binary.other, Node):
            constant_key = "this"
            name = make_as_identifier(f"literal_{binary.this}")
            buffer = torch.tensor(binary.this)
        elif isinstance(binary.other, float) and isinstance(binary.this, Node):
            constant_key = "other"
            name = make_as_identifier(f"literal_{binary.other}")
            buffer = torch.tensor(binary.other)
        else:
            return {}

        with graph.inserting_before(node):
            constant = GetAttr.create(graph, name, buffer)
            args_, kwargs_ = binary.args_kwargs(**{constant_key: constant})
            if replacement := Binary.create_from_overloadpacket(
                graph,
                args=args_,
                kwargs=kwargs_,
                overloadpacket=binary.target.overloadpacket,
            ):
                inject_stack_trace_from(node, to=replacement)
                return {node: ReplaceAllUses(by=replacement.node)}
        graph.erase_node(constant.node)
        return {}


def make_as_identifier(s: str) -> str:
    """Convert a string into a valid Python identifier.

    Args:
        s (str): The input string to convert

    Returns:
        str: A valid Python identifier derived from the input string
    """
    # Remove invalid characters and replace them with underscores
    s = re.sub(r"\W|^(?=\d)", "_", s)

    # If the result is a Python keyword, add a suffix to make it a valid identifier
    if keyword.iskeyword(s):
        s += "_id"

    return s
