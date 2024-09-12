from collections.abc import Callable
from typing import Any

import torch
from torch.export import ExportedProgram
from torch.export._trace import _export as torch_export
from torch.nn.attention import sdpa_kernel

from .arguments_for_export import ArgumentsForExport
from .types import SDPBackend
from .wrappers import PreExportWrapper


def export(
    model: torch.nn.Module,
    arguments: ArgumentsForExport,
    *,
    input_processors: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    output_processors: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    force_static_export: bool = False,
    strict: bool = True,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.MATH,
) -> ExportedProgram:
    with sdpa_kernel(sdp_backends):
        return torch_export(
            PreExportWrapper(
                model,
                input_processors=input_processors,
                output_processors=output_processors,
                constant_inputs=arguments.constant_inputs,
            ),
            args=(),
            kwargs=arguments.tensor_inputs,
            dynamic_shapes=None if force_static_export else {"kwargs": arguments.constraints},
            strict=strict,
            pre_dispatch=pre_dispatch,
        )
