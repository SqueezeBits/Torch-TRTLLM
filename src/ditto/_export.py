import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.export import ExportedProgram
from torch.export._trace import _export as torch_export
from torch.nn.attention import sdpa_kernel
from transformers import PreTrainedModel

from .arguments_for_export import ArgumentsForExport
from .types import SDPBackend
from .wrappers import PreExportWrapper, TRTLLMPreTrainedModelWrapper

logger = logging.getLogger(__name__)


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
    if isinstance(model, PreTrainedModel) and not model._supports_sdpa:
        logger.warning(
            f"{type(model).__name__} doesn't have attention implementation via "
            "`torch.nn.functional.scaled_dot_product_attention`."
        )
    wrapper = TRTLLMPreTrainedModelWrapper if isinstance(model, PreTrainedModel) else PreExportWrapper
    constraints = arguments.constraints if isinstance(model, PreTrainedModel) else {"kwargs": arguments.constraints}
    with sdpa_kernel(sdp_backends):
        exported_program = torch_export(
            wrapper(
                model,
                process_inputs=process_inputs,
                process_outputs=process_outputs,
                constant_inputs=arguments.constant_inputs,
            ),
            args=(),
            kwargs=arguments.tensor_inputs,
            dynamic_shapes=None if force_static_export else constraints,
            strict=strict,
            pre_dispatch=pre_dispatch,
        )
        if isinstance(model, PreTrainedModel):
            exported_program.graph_module.meta["pretrained_config"] = model.config
        return exported_program
