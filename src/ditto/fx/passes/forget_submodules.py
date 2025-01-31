from torch.fx import Node

from ..nodes import GetAttr
from .infra import NodewiseOptimizationPass, NodewisePassResult, ReplaceAllUses, propagate_metadata_from


class ForgetSubmodules(NodewiseOptimizationPass):
    """Forget all nested submodules and unnest all nested parameters."""

    def rewrite(self, node: Node) -> dict[Node, NodewisePassResult]:
        if not (
            (get_attr := GetAttr.specialize_from(node))
            and "." in get_attr.target
            and (graph_module := (graph := node.graph).owning_module)
        ):
            return {}

        def get_qualname() -> str:
            i = 0
            qualname = "constant"
            while hasattr(graph_module, qualname):
                i += 1
                qualname = f"constant_{i}"
            return qualname

        with graph.inserting_before(node):
            unnested_get_attr = GetAttr.create(graph, get_qualname(), get_attr.parameter)
            propagate_metadata_from(get_attr, to=unnested_get_attr)
        return {get_attr.node: ReplaceAllUses(by=unnested_get_attr.node)}
