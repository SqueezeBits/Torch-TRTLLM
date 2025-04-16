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
from awq.modules.linear.gemm import WQLinear_GEMM, WQLinearMMFunction

from ..custom_ops import ditto_dequantize
from .patch import custom_patch


@custom_patch(
    name="awq.modules.linear.gemm.WQLinear_GEMM",
    reason="applying custom dequantize operation",
    required=True,
    env_var_to_disable="DISABLE_AUTO_AWQ_WQLINEAR_GEMM_PATCH",
)
def patch_wqlinear_mm_func_forward() -> None:
    def patched_wqlinear_mm_func_forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
    ):
        out_shape = x.shape[:-1] + (out_features,)

        out = ditto_dequantize(qweight, w_bit, False, scales.dtype, scales, qzeros, group_size)
        out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out

    def patched_gemm_forward(self: WQLinear_GEMM, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)

        with torch.no_grad():
            out = WQLinearMMFunction.apply(
                x,
                self.unpacked_weight,
                self.unpacked_zeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )

        return out.reshape(out_shape)

    WQLinearMMFunction.forward = patched_wqlinear_mm_func_forward
    WQLinear_GEMM.forward = patched_gemm_forward
