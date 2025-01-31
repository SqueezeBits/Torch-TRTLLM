import re

import torch
from loguru import logger
from peft import LoraConfig, PeftModel
from torch.export import ExportedProgram
from torch.export._trace import _export as torch_export
from torch.nn.attention import sdpa_kernel
from transformers import PreTrainedModel

from .arguments.torch_export_arguments import TorchExportArguments
from .constants import PEFT_ADAPTER_PREFIX
from .types import BuiltInConstant, SDPBackend, verify


def export(
    model: PreTrainedModel | PeftModel,
    arguments: TorchExportArguments,
    *,
    strict: bool = False,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.MATH,
) -> ExportedProgram:
    """Export a PyTorch model to an ExportedProgram.

    Args:
        model (PreTrainedModel | PeftModel): The model to export
        arguments (TorchExportArguments): Export arguments containing tensor inputs and constraints
        strict (bool, optional): Whether to run in strict mode. Defaults to False.
        pre_dispatch (bool, optional): Whether to pre-dispatch the model. Defaults to False.
        sdp_backends (SDPBackend | list[SDPBackend], optional): SDP backend(s) to use. Defaults to SDPBackend.MATH.

    Returns:
        ExportedProgram: The exported program
    """
    if not model._supports_sdpa:
        logger.warning(
            f"{type(model).__name__} doesn't have attention implementation via "
            "`torch.nn.functional.scaled_dot_product_attention`."
        )

    with sdpa_kernel(sdp_backends):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(model.dtype in (torch.float16, torch.bfloat16))
        exported_program = torch_export(
            ConstantInputFilterer(model, constant_inputs=arguments.constant_inputs),
            args=(),
            kwargs=arguments.tensor_inputs,
            dynamic_shapes={"kwargs": arguments.constraints},
            strict=strict,
            pre_dispatch=pre_dispatch,
        )
        try:
            exported_program.graph_module.meta["pretrained_config"] = model.config
        except AttributeError:
            logger.warning("model.config is not available")

        if isinstance(model, PeftModel):
            try:
                if peft_configs := verify(model.base_model.peft_config, as_type=dict[str, LoraConfig]):
                    exported_program.graph_module.meta["peft_configs"] = {
                        int(match.group(1)): config
                        for key, config in peft_configs.items()
                        if (match := re.match(rf"^{PEFT_ADAPTER_PREFIX}_(\d+)$", key)) is not None
                    }
            except AttributeError:
                logger.warning("model.base_model.peft_config is not available")
        return exported_program


class ConstantInputFilterer(torch.nn.Module):
    """Module that filters constant inputs from model forward arguments.

    Args:
        model (torch.nn.Module): The model to wrap
        constant_inputs (dict[str, BuiltInConstant] | None, optional): Constant inputs to pass to model.
            Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        constant_inputs: dict[str, BuiltInConstant] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.constants = constant_inputs or {}

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Forward pass that merges constant and tensor inputs.

        Args:
            **kwargs (torch.Tensor): Tensor inputs to pass to model

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]: Model outputs
        """
        return self.model(**kwargs, **self.constants)
