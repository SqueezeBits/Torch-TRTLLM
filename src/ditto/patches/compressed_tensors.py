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
from compressed_tensors.compressors.quantized_compressors.pack_quantized import PackedQuantizationCompressor
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize

from .patch import custom_patch


@custom_patch(
    name="compressed_tensors.compressors.quantized_compressors.pack_quantized.PackedQuantizationCompressor",
    reason="",
    required=True,
    env_var_to_disable="DISABLE_COMPRESSED_TENSORS_PACKED_QUANTIZATION_COMPRESSOR_PATCH",
)
def patch_decompress_weight() -> None:
    def patched_decompress_weight(
        self,
        compressed_data: dict[str, torch.Tensor],
        quantization_args: QuantizationArgs | None = None,
    ) -> torch.Tensor:
        weight = compressed_data["weight_packed"]
        scale = compressed_data["weight_scale"]
        zero_point = compressed_data.get("weight_zero_point", None)
        g_idx = compressed_data.get("weight_g_idx", None)
        num_bits = quantization_args.num_bits

        shifts = torch.arange(0, 32, num_bits, device=weight.device)
        unpacked = torch.bitwise_right_shift(weight[:, :, None], shifts[None, None, :]).to(torch.int32)
        unpacked = unpacked.view(unpacked.shape[0], -1) - (pow(2, num_bits) // 2)
        unpacked = unpacked.to(torch.int8)

        decompressed_weight = dequantize(x_q=unpacked, scale=scale, zero_point=zero_point, g_idx=g_idx)

        return decompressed_weight

    PackedQuantizationCompressor.decompress_weight = patched_decompress_weight
