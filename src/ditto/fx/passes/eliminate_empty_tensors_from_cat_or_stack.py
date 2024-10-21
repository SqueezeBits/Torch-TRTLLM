import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.shape_prop import TensorMetadata
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications


def eliminate_empty_tensors_from_cat_or_stack(graph_module: GraphModule) -> GraphModule:
    for node in graph_module.graph.nodes:
        if not (
            node.target in (torch.ops.aten.cat.default, torch.ops.aten.stack.default)
            and isinstance((tensors := node.args[0]), list | tuple)
            and all(isinstance(tensor, Node) for tensor in tensors)
        ):
            continue
        non_empty_tensors = tuple(
            tensor
            for tensor in tensors
            if isinstance(tensor, Node)
            and isinstance((tensor_meta := tensor.meta.get("tensor_meta", None)), TensorMetadata)
            and tensor_meta.shape.numel() > 0
        )
        node.args = (non_empty_tensors,) + node.args[1:]
    clean_up_graph_after_modifications(graph_module)
    return graph_module
