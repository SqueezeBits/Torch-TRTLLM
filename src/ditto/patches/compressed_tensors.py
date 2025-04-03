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
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle import forward

from .patch import custom_patch


@custom_patch(
    name="compressed_tensors.compressors.quantized_compressors.pack_quantized.PackedQuantizationCompressor",
    reason="resolving torch.export error due to unsupported operation.",
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
        decompressed_weight = forward.dequantize(x_q=unpacked, scale=scale, zero_point=zero_point, g_idx=g_idx)

        return decompressed_weight

    PackedQuantizationCompressor.decompress_weight = patched_decompress_weight


@custom_patch(
    name="compressed linear process",
    reason="resolving torch.export error and the registration of the parameters during forward pass.",
    required=True,
    env_var_to_disable="DISABLE_COMPRESSED_TENSORS_COMPRESSED_LINEAR_PROCESS_PACH",
)
def patch_compressed_linear_process() -> None:
    def patched_forward(self, input: torch.Tensor) -> torch.Tensor:
        unpacked_weight = self.compressor.decompress_module(self)
        return torch.nn.functional.linear(input, unpacked_weight, self.bias)

    def patched_process_quantization(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        args: QuantizationArgs,
        g_idx: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        do_quantize: bool = True,
        do_dequantize: bool = True,
    ) -> torch.Tensor:
        q_min, q_max = forward.calculate_range(args, x.device)
        group_size = args.group_size

        if args.strategy == QuantizationStrategy.GROUP:
            while scale.ndim < 2:
                # pad scale and zero point dims for slicing
                scale = scale.unsqueeze(1)
                zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

            assert g_idx is None or -1 in g_idx
            assert zero_point is None, "zero point is not supported yet"
            scale = scale.reshape(scale.shape[0], 1, -1)
            output = x.reshape(x.shape[0], group_size, -1)

            if do_dequantize:
                output = forward._dequantize(output, scale, zero_point)

            output = output.reshape(output.shape[0], output.shape[1] * output.shape[2])

        else:  # covers channel, token and tensor strategies
            if do_quantize:
                output = forward._quantize(
                    x,
                    scale,
                    zero_point,
                    q_min,
                    q_max,
                    args,
                    dtype=dtype,
                )
            if do_dequantize:
                output = forward._dequantize(output if do_quantize else x, scale, zero_point)

        return output

    forward._process_quantization = patched_process_quantization
    CompressedLinear.forward = patched_forward
