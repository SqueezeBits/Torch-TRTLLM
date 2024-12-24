from collections.abc import Callable
from inspect import isclass

from loguru import logger
from pydantic import Field
from torch.fx import GraphModule

from ....constants import FX_TRANSFORM_MAXIMUM_ITERATION
from ....debug import get_memory_footprint
from ....types import StrictlyTyped
from .graph_pass import GraphOptimizationPass
from .pass_result import PassResult

PassType = Callable[[GraphModule], PassResult] | GraphOptimizationPass


class PassManager(StrictlyTyped):
    steps: int = Field(default=FX_TRANSFORM_MAXIMUM_ITERATION, ge=0)
    passes: list[PassType] = Field(default_factory=list)
    warn_on_partial_convergence: bool = True

    def __call__(self, graph_module: GraphModule) -> PassResult:
        logger.opt(lazy=True).trace("Memory Footprint: {m}", m=get_memory_footprint)
        overall_modified = False
        for step in range(self.steps):
            modified = False
            logger.debug(f"Running iteration {step + 1}")
            # pylint: disable-next=not-an-iterable
            for p in self.passes:
                res = p(graph_module)
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
        if isclass(p):
            assert issubclass(p, GraphOptimizationPass)
            self.passes.append(p())
            return
        self.passes.append(p)  # type: ignore[arg-type]

    def as_transform(self) -> Callable[[GraphModule], GraphModule]:
        return lambda graph_module: self(graph_module).graph_module
