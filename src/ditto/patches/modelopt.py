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
from modelopt.torch.quantization.nn.modules.quant_module import QuantLinearConvBase

from ..custom_ops import ditto_fake_quantize
from .patch import custom_patch


@custom_patch(
    name="modelopt.torch.quantization.nn.modules.quant_module.QuantLinearConvBase",
    reason="applying custom dequantize operation",
    required=True,
    env_var_to_disable="DISABLE_MODELOPT_QUANT_LINEAR_CONV_BASE_PATCH",
)
def patch_quantize_forward() -> None:
    def patched_forward(self: QuantLinearConvBase, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.input_quantizer.is_enabled:
            inp = ditto_fake_quantize(
                inp,
                8,
                self.input_quantizer._dynamic,
                inp.dtype,
                self.input_scale if self.input_quantizer._dynamic is False else None,
                None,
                None,
                getattr(self.input_quantizer, "pre_quant_scale", None),
            )
        if self.weight_quantizer.is_enabled:
            weight = ditto_fake_quantize(
                self.unpacked_weight,
                8 if self.weight_quantizer.num_bits == (4, 3) else self.weight_quantizer.num_bits,
                self.weight_quantizer._dynamic,
                self.scales.dtype,
                self.scales,
                self.unpacked_zeros,
                self.weight_quantizer.block_sizes.get(-1, None) if self.weight_quantizer.block_sizes else None,
            ).T
        else:
            weight = self.weight

        out = torch.nn.functional.linear(inp, weight, self.bias if self.bias is not None else None)

        if self.output_quantizer.is_enabled:
            out = ditto_fake_quantize(
                out,
                8 if self.output_quantizer.num_bits == (4, 3) else self.output_quantizer.num_bits,
                self.output_quantizer._dynamic,
                out.dtype,
                self.output_scale if self.output_quantizer._dynamic is False else None,
            )

        return out

    QuantLinearConvBase.forward = patched_forward
