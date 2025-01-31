from torch.fx import GraphModule

from ...debug import save_for_debug
from ..nodes import GPTAttention
from ..subgraphs import Linear
from .infra import GraphOptimizationPass, PassResult


class IndexLayers(GraphOptimizationPass):
    """Index mm nodes by their layer index in the graph."""

    def postprocess(self, graph_module: GraphModule) -> None:
        super().postprocess(graph_module)
        save_for_debug("after_index_layers", graph_module)

    def call(self, graph_module: GraphModule) -> PassResult:
        layer_index: int | None = None
        for node in graph_module.graph.nodes:
            if gpt_attention := GPTAttention.specialize_from(node):
                layer_index = gpt_attention.target.layer_idx
                if qkv_proj := Linear.configure_from(gpt_attention.qkv):
                    qkv_proj.layer_index = layer_index
                continue

            if layer_index is None or (linear := Linear.configure_from(node)) is None:
                continue

            linear.layer_index = layer_index
        return PassResult(graph_module=graph_module, modified=False)
