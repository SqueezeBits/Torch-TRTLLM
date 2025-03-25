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
from awq.modules.linear.gemm import WQLinearMMFunction
from awq.utils.packing_utils import dequantize_gemm

from .patch import custom_patch


@custom_patch(
    name="awq.modules.linear.gemm.WQLinearMMFunction.forward",
    reason="",
    required=True,
    env_var_to_disable="DISABLE_AUTO_AWQ_WQLINEARMMFUNCTION_PATCH",
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
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)

        out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
        out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out

    WQLinearMMFunction.forward = patched_wqlinear_mm_func_forward
