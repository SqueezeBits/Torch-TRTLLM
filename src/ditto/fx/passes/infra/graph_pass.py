from abc import ABC, abstractmethod
from collections.abc import Callable

from loguru import logger
from torch.fx import GraphModule, Node

from ....debug import get_memory_footprint
from ....types import StrictlyTyped
from .cleanup import cleanup
from .pass_result import PassResult


class GraphOptimizationPass(StrictlyTyped, ABC):
    depth: int = 0
    register_create_node_hook: bool = True

    @property
    def indent(self) -> str:
        return " " * (2 * self.depth)

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def file(self) -> str:
        return f"{type(self).__module__.replace('.', '/')}.py"

    def create_stack_trace(self, node: Node) -> None:
        node.stack_trace = f'File "{self.file}", line 0, in call\n    Created by {self.name}'

    def preprocess(self, graph_module: GraphModule) -> None:
        if self.register_create_node_hook:
            graph_module._register_create_node_hook(self.create_stack_trace)

    def postprocess(self, graph_module: GraphModule) -> None:
        if self.register_create_node_hook:
            graph_module._unregister_create_node_hook(self.create_stack_trace)

    @abstractmethod
    def call(self, graph_module: GraphModule) -> PassResult:
        ...

    def __call__(self, graph_module: GraphModule) -> PassResult:
        self.preprocess(graph_module)
        logger.debug(f"{self.indent}Running pass {type(self).__name__}")
        result = self.call(graph_module)
        if result is not None:
            logger.debug(f"{self.indent}-> modified" if result.modified else f"{self.indent}-> no changes")
            if result.modified:
                cleanup(graph_module, run_fake_tensor_prop=result.require_fake_tensor_prop)
        else:
            logger.debug(f"{self.indent}-> no result returned")
        self.postprocess(graph_module)
        logger.opt(lazy=True).trace("Memory Footprint: {m}", m=get_memory_footprint)
        return result

    def as_transform(self) -> Callable[[GraphModule], GraphModule]:
        return lambda graph_module: self(graph_module).graph_module
