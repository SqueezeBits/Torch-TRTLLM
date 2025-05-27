# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Callable

from loguru import logger
from torch.fx import GraphModule, Node
from torch_tensorrt.dynamo import CompilationSettings

from ....debug import get_memory_footprint
from ....types import StrictlyTyped
from .cleanup import cleanup
from .pass_result import PassResult


class GraphOptimizationPass(StrictlyTyped, ABC):
    """The base class for graph optimization passes.

    Attributes:
        depth (int): The depth of the pass.
        register_create_node_hook (bool): Whether to register the create node hook.
    """

    depth: int = 0
    register_create_node_hook: bool = True

    @property
    def indent(self) -> str:
        """The indent for the debug messages."""
        return " " * (2 * self.depth)

    @property
    def name(self) -> str:
        """The name of the pass."""
        return type(self).__name__

    @property
    def file(self) -> str:
        """The file of the pass."""
        return f"{type(self).__module__.replace('.', '/')}.py"

    def create_stack_trace(self, node: Node) -> None:
        """Create the stack trace for the node.

        Args:
            node (Node): The node to create the stack trace for.
        """
        node.stack_trace = f'File "{self.file}", line 0, in call\n    Created by {self.name}'

    def preprocess(self, graph_module: GraphModule) -> None:
        """Preprocess the graph module.

        Args:
            graph_module (GraphModule): The graph module to preprocess.
        """
        if self.register_create_node_hook:
            graph_module._register_create_node_hook(self.create_stack_trace)

    def postprocess(self, graph_module: GraphModule) -> None:
        """Postprocess the graph module.

        Args:
            graph_module (GraphModule): The graph module to postprocess.
        """
        if self.register_create_node_hook:
            graph_module._unregister_create_node_hook(self.create_stack_trace)

    @abstractmethod
    def call(self, graph_module: GraphModule) -> PassResult:
        """Process the main logic of the pass.

        Args:
            graph_module (GraphModule): The graph module to transform.

        Returns:
            PassResult: The result of the pass.
        """

    def __call__(self, graph_module: GraphModule, settings: CompilationSettings) -> PassResult:
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

    def as_transform(self) -> Callable[[GraphModule, CompilationSettings], GraphModule]:
        """Convert the pass to a callable that takes a graph module and returns the transformed graph module.

        Returns:
            Callable[[GraphModule, CompilationSettings], GraphModule]: The callable that takes a graph module and
                returns the transformed graph module.
        """
        return lambda graph_module, settings: self(graph_module, settings).graph_module
