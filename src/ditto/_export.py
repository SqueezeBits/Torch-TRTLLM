

from typing import Any

import torch
from torch.export._trace import _export as torch_export
from torch.export.dynamic_shapes import _Dim as DimType

from .cache_handler import CacheHandler
from .wrappers import PostExportWrapper, PreExportWrapper


def export(
    cache_handler: CacheHandler,
    model: torch.nn.Module,
    example_inputs: dict[str, Any],
    dynamic_shapes: dict[str, dict[int, DimType] | None],
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
    maintain_input_constraints_checking: bool = False,
) -> torch.nn.Module:
    graph_module = torch_export(
        PreExportWrapper(model, cache_handler=cache_handler),
        (),
        example_inputs,
        dynamic_shapes={"kwargs": dynamic_shapes},
        strict=strict,
        pre_dispatch=pre_dispatch,
    ).module()
    if not maintain_input_constraints_checking:
        graph_module._forward_pre_hooks.clear()
    return PostExportWrapper(graph_module, cache_handler=cache_handler)
