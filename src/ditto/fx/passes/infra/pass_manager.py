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

from collections.abc import Callable
from inspect import isclass

from loguru import logger
from pydantic import Field
from torch.fx import GraphModule
from torch_tensorrt.dynamo import CompilationSettings

from ....constants import FX_TRANSFORM_MAXIMUM_ITERATION
from ....debug import get_memory_footprint
from ....types import StrictlyTyped
from .graph_pass import GraphOptimizationPass
from .pass_result import PassResult

PassType = Callable[[GraphModule, CompilationSettings], PassResult] | type[GraphOptimizationPass]


class PassManager(StrictlyTyped):
    """The manager for the passes.

    Attributes:
        steps (int): The number of steps to run the passes.
        passes (list[PassType]): The passes to run.
        warn_on_partial_convergence (bool): Whether to warn on partial convergence.
    """

    steps: int = Field(default=FX_TRANSFORM_MAXIMUM_ITERATION, ge=0)
    passes: list[PassType] = Field(default_factory=list)
    warn_on_partial_convergence: bool = True

    def __call__(self, graph_module: GraphModule, settings: CompilationSettings | None = None) -> PassResult:
        logger.opt(lazy=True).trace("Memory Footprint: {m}", m=get_memory_footprint)
        overall_modified = False
        for step in range(self.steps):
            modified = False
            logger.debug(f"Running iteration {step + 1}")
            # pylint: disable-next=not-an-iterable
            for p in self.passes:
                res = p(graph_module, settings or CompilationSettings())
                graph_module = res.graph_module
                modified = modified or res.modified
            overall_modified = overall_modified or modified
            if not modified:
                logger.debug(f"GraphModule converged after {step + 1} iterations")
                break
        else:
            if self.warn_on_partial_convergence:
                logger.warning(
                    f"GraphModule has not fully converged after {self.steps} iterations of pass loops. "
                    f"Set the environment variable FX_TRANSFORM_MAXIMUM_ITERATION to a value larger than {self.steps}"
                    " for a full convergence."
                )
        return PassResult(graph_module=graph_module, modified=overall_modified)

    def add_pass(self, p: PassType | type[GraphOptimizationPass]) -> None:
        """Add a pass to the manager.

        Args:
            p (PassType | type[GraphOptimizationPass]): The pass to add.
        """
        if isclass(p):
            assert issubclass(p, GraphOptimizationPass)
            self.passes.append(p())
            return
        self.passes.append(p)

    def as_transform(self) -> Callable[[GraphModule, CompilationSettings], GraphModule]:
        """Convert the manager to a callable that applies the passes to a graph module.

        Returns:
            Callable[[GraphModule, CompilationSettings], GraphModule]: The callable that takes a graph module
                and returns the transformed graph module.
        """
        return lambda graph_module, settings: self(graph_module, settings).graph_module
