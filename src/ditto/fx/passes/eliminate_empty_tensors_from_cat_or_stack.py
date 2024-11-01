from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult

from ..utils import get_tensor_metadata
from .graph_pass import GraphOptimizationPass
from .specialized_node import CatNode, StackNode


class EliminateEmptyTensorsFromCatOrStack(GraphOptimizationPass):
    """Reuse constant attributes sharing the same values."""

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if not (
                (cat_or_stack := CatNode.specialize_from(node) or StackNode.specialize_from(node))
                and (tensor_metas := [meta for t in cat_or_stack.tensors if (meta := get_tensor_metadata(t))])
                and len(tensor_metas) == len(cat_or_stack.tensors)
            ):
                continue
            non_empty_tensors = tuple(
                tensor
                for tensor_meta, tensor in zip(tensor_metas, cat_or_stack.tensors)
                if tensor_meta.shape.numel() > 0
            )
            if len(non_empty_tensors) == len(cat_or_stack.tensors):
                continue
            node.args = (non_empty_tensors,) + node.args[1:]
            modified = True
        return PassResult(graph_module, modified)
