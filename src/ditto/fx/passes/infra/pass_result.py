from torch.fx import GraphModule

from ....types import StrictlyTyped


class PassResult(StrictlyTyped):
    graph_module: GraphModule
    modified: bool
    require_fake_tensor_prop: bool = False
