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
from torch._decomp import remove_decompositions
from torch_tensorrt.dynamo.lowering._decomposition_groups import TORCH_TRT_DECOMPOSITIONS
from torch_tensorrt.dynamo.lowering._decompositions import register_torch_trt_decomposition
from torch_tensorrt.fx.converters import converter_utils

from .patch import custom_patch


# Note: These patches are temporary patches to resolve the issue.
# These will be removed once the issue is resolved in torch_tensorrt.
@custom_patch(
    name="patch_torch_tensorrt_issues",
    reason="resolving the issue about torch_tensorrt issues",
    required=True,
    env_var_to_disable="DISABLE_TORCH_TENSORRT_ISSUES_PATCH",
)
def patch_torch_tensorrt_issues() -> None:
    # https://github.com/pytorch/TensorRT/pull/3563
    def patched_converter_utils_type_cast(network, target, name: str, input, cast_type):
        layer_i = network.add_cast(input, cast_type)
        converter_utils.set_layer_name(layer_i, target, f"{name}_dtype_change")
        return layer_i.get_output(0)

    remove_decompositions(TORCH_TRT_DECOMPOSITIONS, [torch.ops.aten.full_like])

    # https://github.com/pytorch/TensorRT/pull/3535
    @register_torch_trt_decomposition(torch.ops.aten.full_like, registry=TORCH_TRT_DECOMPOSITIONS)  # type: ignore
    def full_like_decomposition(*args, **kwargs) -> torch.Tensor:
        input = args[0]
        shape = args[0].shape
        fill_value = args[1]
        kwargs["dtype"] = kwargs.get("dtype", None) or input.dtype
        kwargs["device"] = kwargs.get("device", None) or input.device
        return torch.full(shape, fill_value, dtype=kwargs["dtype"], device=kwargs["device"])

    converter_utils.type_cast = patched_converter_utils_type_cast
