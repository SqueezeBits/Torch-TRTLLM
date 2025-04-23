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

import warnings

import torch

from ..custom_ops import ditto_fake_quantize
from .patch import custom_patch

warnings.simplefilter("ignore", FutureWarning)

from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear  # noqa: E402
from auto_gptq.utils import import_utils  # noqa: E402


@custom_patch(
    name="auto_gptq.nn_modules.qlinear.qlinear_cuda_old.QuantLinear",
    reason="applying custom fake quantize operation",
    required=True,
    env_var_to_disable="DISABLE_AUTO_GPTQ_QUANTLINEAR_PATCH",
)
def patch_dynamically_import_quantlinear() -> None:
    def patched_dynamically_import_quantlinear(
        use_triton: bool,
        desc_act: bool,
        group_size: int,
        bits: int,
        disable_exllama: bool | None = None,
        disable_exllamav2: bool = False,
        use_qigen: bool = False,
        disable_marlin: bool = True,
    ):
        return QuantLinear

    def patched_forward(self: QuantLinear, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.outfeatures,)
        weight = ditto_fake_quantize(
            self.unpacked_weight,
            self.bits,
            False,
            self.scales.dtype,
            self.scales,
            self.unpacked_zeros,
            self.group_size,
        )
        out = torch.matmul(x, weight)
        out = out.to(dtype=x.dtype).reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    import_utils.dynamically_import_QuantLinear = patched_dynamically_import_quantlinear
    QuantLinear.forward = patched_forward
