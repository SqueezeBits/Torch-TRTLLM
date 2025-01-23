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

import torch
from loguru import logger
from torch.export import ExportedProgram
from torch.export._trace import _export as torch_export
from torch.nn.attention import sdpa_kernel
from transformers import PreTrainedModel

from .arguments.torch_export_arguments import TorchExportArguments
from .types import BuiltInConstant, SDPBackend


def export(
    model: PreTrainedModel,
    arguments: TorchExportArguments,
    *,
    strict: bool = False,
    pre_dispatch: bool = False,
    sdp_backends: SDPBackend | list[SDPBackend] = SDPBackend.MATH,
) -> ExportedProgram:
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
