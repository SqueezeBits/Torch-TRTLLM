import torch
from loguru import logger
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch_tensorrt.dynamo.lowering.passes.pass_utils import clean_up_graph_after_modifications

from ...constants import DEFAULT_DEVICE


class GraphOptimizationPass(PassBase):
    def __init__(self, *, depth: int = 0) -> None:
        super().__init__()
        self.depth = depth

    @property
    def indent(self) -> str:
        return " " * (2 * self.depth)

    def __call__(self, graph_module: GraphModule) -> PassResult | None:
        self.requires(graph_module)
        logger.debug(f"{self.indent}Running pass {type(self).__name__}")
        result = self.call(graph_module)
        if result is not None:
            logger.debug(f"{self.indent}-> modified" if result.modified else f"{self.indent}-> no changes")
            if result.modified:
                clean_up_graph_after_modifications(graph_module)
        else:
            logger.debug(f"{self.indent}-> no result returned")
        self.ensures(graph_module)
        log_memory_footprint(graph_module)
        return result


def log_memory_footprint(graph_module: GraphModule) -> None:
    try:
        device = next(iter(graph_module.parameters())).device
    except StopIteration:
        device = torch.device(DEFAULT_DEVICE)

    logger.opt(lazy=True).trace(
        "Allocated Memory: {x:.2f} MB", x=lambda: torch.cuda.memory_allocated(device) / 1024**2
    )
    logger.opt(lazy=True).trace(
        "Max Allocated Memory: {x:.2f} MB", x=lambda: torch.cuda.max_memory_allocated(device) / 1024**2
    )
    logger.opt(lazy=True).trace("Cached Memory: {x:.2f} MB", x=lambda: torch.cuda.memory_reserved(device) / 1024**2)
