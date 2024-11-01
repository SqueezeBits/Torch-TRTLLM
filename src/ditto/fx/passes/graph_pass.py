from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications


class GraphOptimizationPass(PassBase):
    def __init__(self, depth: int = 0) -> None:
        super().__init__()
        self.depth = depth

    @property
    def indent(self) -> str:
        return " " * (2 * self.depth)

    def __call__(self, graph_module: GraphModule) -> PassResult | None:
        print(f"{self.indent}Running pass {type(self).__name__}")
        result = super().__call__(graph_module)
        if result is not None:
            print(f"{self.indent}-> modified" if result.modified else f"{self.indent}-> no changes")
            if result.modified:
                clean_up_graph_after_modifications(graph_module)
        else:
            print(f"{self.indent}-> no result returned")
        return result
