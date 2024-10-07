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
    process_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    process_outputs: Callable[[Any], Any] | None = None,
    force_static_export: bool = False,
    strict: bool = False,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.MATH,
) -> ExportedProgram:
    with sdpa_kernel(sdp_backends):
        return torch_export(
            PreExportWrapper(
                model,
                process_inputs=process_inputs,
                process_outputs=process_outputs,
                constant_inputs=arguments.constant_inputs,
            ),
            args=(),
            kwargs=arguments.tensor_inputs,
            dynamic_shapes=None if force_static_export else {"kwargs": arguments.constraints},
            strict=strict,
            pre_dispatch=pre_dispatch,
        )
