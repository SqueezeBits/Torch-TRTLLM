import torch
from loguru import logger
from torch.export import ExportedProgram
from torch.export._trace import _export as torch_export
from torch.nn.attention import sdpa_kernel
from transformers import PreTrainedModel

from .arguments.torch_export_arguments import TorchExportArguments
from .contexts import disable_torch_jit_state
from .types import BuiltInConstant, SDPBackend


def export(
    model: PreTrainedModel,
    arguments: TorchExportArguments,
    *,
    strict: bool = False,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.EFFICIENT_ATTENTION,
) -> ExportedProgram:
    if not model._supports_sdpa:
        logger.warning(
            f"{type(model).__name__} doesn't have attention implementation via "
            "`torch.nn.functional.scaled_dot_product_attention`."
        )

    with sdpa_kernel(sdp_backends), disable_torch_jit_state():
        exported_program = torch_export(
            ConstantInputFilterer(model, constant_inputs=arguments.constant_inputs),
            args=(),
            kwargs=arguments.tensor_inputs,
            dynamic_shapes={"kwargs": arguments.constraints},
            strict=strict,
            pre_dispatch=pre_dispatch,
        )
        if isinstance(model, PreTrainedModel):
            exported_program.graph_module.meta["pretrained_config"] = model.config
        return exported_program


class ConstantInputFilterer(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        constant_inputs: dict[str, BuiltInConstant] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.constants = constant_inputs or {}

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        return self.model(**kwargs, **self.constants)
