import torch
from torch.fx import GraphModule, Node
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications


def eliminate_nop_cat_or_stack(graph_module: GraphModule) -> GraphModule:
    for node in graph_module.graph.nodes:
        if not (
            node.target in (torch.ops.aten.cat.default, torch.ops.aten.stack.default)
            and isinstance((tensors := node.args[0]), list | tuple)
            and len(tensors) == 1
            and isinstance((the_input := tensors[0]), Node)
        ):
            continue
        node.replace_all_uses_with(the_input)
    clean_up_graph_after_modifications(graph_module)
    return graph_module
