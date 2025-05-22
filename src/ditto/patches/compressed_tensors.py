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

from collections import OrderedDict
from copy import deepcopy

import torch
from compressed_tensors import quantization
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import forward
from compressed_tensors.quantization.lifecycle.apply import (
    find_name_or_class_matches,
    fix_fsdp_module_name,
    iter_named_quantizable_modules,
    process_quantization_config,
)
from compressed_tensors.quantization.lifecycle.initialize import _initialize_attn_scales, is_attention_module
from compressed_tensors.quantization.utils.helpers import KV_CACHE_TARGETS

from ..torch.ops import ditto_fake_quantize
from .patch import custom_patch


@custom_patch(
    name="compressed_tensors.linear.compressed_linear.CompressedLinear",
    reason="resolving torch.export error and the registration of the parameters "
    "and applying custom fake quantize operation",
    required=True,
    env_var_to_disable="DISABLE_COMPRESSED_TENSORS_COMPRESSED_LINEAR_PROCESS_PATCH",
)
def patch_compressed_linear_process() -> None:
    original_from_linear = CompressedLinear.from_linear

    @torch.no_grad()
    def patched_from_linear(module: torch.nn.Linear, quantization_scheme: QuantizationScheme, quantization_format: str):
        ret = original_from_linear(module, quantization_scheme, quantization_format)

        if (
            quantization_scheme.weights
            and not quantization_scheme.weights.symmetric
            and (weight_zero_point := getattr(ret, "weight_zero_point", None)) is not None
            and quantization_scheme.weights.strategy
            and quantization_scheme.weights.strategy == QuantizationStrategy.GROUP
            and quantization_scheme.weights.group_size
        ):
            expected_shape = (
                weight_zero_point.shape[0] // (32 // quantization_scheme.weights.num_bits),
                weight_zero_point.shape[1],
            )
            new_zero_point = torch.nn.Parameter(
                torch.zeros(expected_shape, device=weight_zero_point.device, dtype=torch.int32), requires_grad=False
            )
            ret.register_parameter("weight_zero_point", new_zero_point)

        return ret

    def patched_forward(self, input: torch.Tensor) -> torch.Tensor:
        unpacked_weight = ditto_fake_quantize(
            self.unpacked_weight,
            self.quantization_scheme.weights.num_bits,
            False,
            self.scales.dtype,
            self.scales,
            self.unpacked_zeros,
            self.quantization_scheme.weights.group_size,
        )
        out = torch.nn.functional.linear(input, unpacked_weight.T, self.bias if self.bias is not None else None)

        if hasattr(self, "output_scale"):
            out = ditto_fake_quantize(out, 8, False, self.output_scale.dtype, self.output_scale)

        return out

    @torch.no_grad()
    def patched_forward_quantize(
        module: torch.nn.Module, value: torch.Tensor, base_name: str, args: QuantizationArgs
    ) -> torch.Tensor:
        if module.quantization_status == QuantizationStatus.COMPRESSED and base_name == "weight":
            return value

        if args.dynamic:
            out = ditto_fake_quantize(value, args.num_bits, True, value.dtype)
        else:
            out = ditto_fake_quantize(
                value,
                args.num_bits,
                False,
                module.input_scale.dtype if base_name == "input" else module.scales.dtype,
                module.input_scale if base_name == "input" else module.scales,
                module.unpacked_zeros,
                args.group_size,
            )

        return out

    origin_apply_quantization_config = quantization.apply_quantization_config

    def patched_apply_quantization_config(
        model: torch.nn.Module, config: QuantizationConfig | None, run_compressed: bool = False
    ) -> OrderedDict:
        if config is None:
            return OrderedDict()

        config = deepcopy(org_config := config)
        target_to_scheme = OrderedDict()
        config = process_quantization_config(config)
        for scheme in config.config_groups.values():
            for target in scheme.targets:
                target_to_scheme[target] = scheme

        attn_name_to_submodule: dict[str, torch.nn.Module] = {}
        for name, submodule in iter_named_quantizable_modules(model, include_children=True, include_attn=True):
            name = fix_fsdp_module_name(name)
            targets = find_name_or_class_matches(name, submodule, target_to_scheme)

            if targets == KV_CACHE_TARGETS and is_attention_module(submodule):
                attn_name_to_submodule[name] = submodule
                _initialize_attn_scales(submodule)

        return origin_apply_quantization_config(model, org_config, run_compressed)

    CompressedLinear.from_linear = patched_from_linear
    CompressedLinear.forward = patched_forward
    forward.forward_quantize = patched_forward_quantize
    quantization.apply_quantization_config = patched_apply_quantization_config
