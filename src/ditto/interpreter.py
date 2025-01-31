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

import json
from collections.abc import Sequence
from typing import Any

import tensorrt as trt
import torch
import torch.fx
from loguru import logger
from torch.fx import GraphModule
from torch.fx.node import Node, Target
from torch.utils._python_dispatch import _disable_current_modes
from torch_tensorrt import Input, dtype
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo.conversion import (
    DYNAMO_CONVERTERS,
    CallingConvention,
    ConversionContext,
    TRTInterpreter,
    UnsupportedOperatorException,
)
from torch_tensorrt.logging import TRT_LOGGER

from .configs import TensorRTBuilderConfig, TensorRTNetworkCreationFlags
from .debug import (
    builder_config_as_dict,
    open_debug_artifact,
    save_for_debug,
)
from .fx.targets import Plugin
from .fx.utils import get_tensor_metadata
from .types import DataType


# pylint: disable-next=too-many-instance-attributes
class TRTLLMInterpreter(TRTInterpreter):
    """TensorRT-LLM interpreter for converting PyTorch models to TensorRT engines.

    Args:
        module (GraphModule): The PyTorch graph module to convert
        input_specs (tuple[Input, ...]): Input specifications for the network
        network_flags (TensorRTNetworkCreationFlags): Network creation flags
        builder_config (TensorRTBuilderConfig): TensorRT builder configuration
        rank (int): Rank of the current process
        engine_cache (BaseEngineCache | None, optional): Cache for TensorRT engines. Defaults to None.
        network_name (str | None, optional): Name of the network. Defaults to None.
        output_names (list[str] | None, optional): Names of output tensors. Defaults to None.
    """

    def __init__(
        self,
        module: GraphModule,
        input_specs: tuple[Input, ...],
        *,
        network_flags: TensorRTNetworkCreationFlags,
        builder_config: TensorRTBuilderConfig,
        rank: int,
        engine_cache: BaseEngineCache | None = None,
        network_name: str | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            module,
            input_specs=input_specs,
            output_dtypes=infer_module_output_dtypes(module),
            compilation_settings=builder_config.get_compilation_settings(),
            engine_cache=engine_cache,
        )
        self.builder = trt.Builder(TRT_LOGGER)
        self.ctx = ConversionContext(self.builder.create_network(network_flags.bitmask), self.compilation_settings)
        self._builder_config = builder_config
        if self.optimization_profiles:
            self._builder_config.optimization_profiles = self.optimization_profiles
        self.logger = logger
        self.placeholder_names = [n.name for n in module.graph.find_nodes(op="placeholder")]
        self.user_output_names = output_names
        self.rank = rank
        if network_name:
            self.ctx.net.name = network_name
        # Note that the `trt.IConstantLayer` created from a graph module's weight holds the data pointer
        # to the corresponding PyTorch tensor, which must be cached along with TensorRT tensors.
        self._constant_cache: dict[str, tuple[trt.ITensor, torch.Tensor]] = {}

    def validate_conversion(self) -> set[str]:
        missing_ops = super().validate_conversion()
        return {missing_op for missing_op in missing_ops if "gpt_attention_plugin" not in missing_op}

    def _construct_trt_network_def(self) -> None:
        super()._construct_trt_network_def()
        save_for_debug(f"trt_network_def_rank{self.rank}", self.ctx.net, self.optimization_profiles)

    def _populate_trt_builder_config(
        self,
        strict_type_constraints: bool = False,
        algorithm_selector: trt.IAlgorithmSelector | None = None,
        tactic_sources: int | None = None,
    ) -> trt.IBuilderConfig:
        builder_config = self.builder.create_builder_config()
        self._builder_config.copy_to(builder_config)
        with open_debug_artifact(f"builder_config_rank{self.rank}.json") as f:
            if f:
                json.dump(builder_config_as_dict(builder_config), f, indent=2, sort_keys=True)
        return builder_config

    def run_node(self, n: Node) -> trt.ITensor | Sequence[trt.ITensor]:
        self.logger.debug(f"Converting {n.format_node(self.placeholder_names) or str(n)}")
        output: trt.ITensor | Sequence[trt.ITensor] = super().run_node(n)
        self.logger.debug(f"{n.name} -> {_format_output(output)}")
        return output

    def call_function(self, target: Target, args: Any, kwargs: Any) -> Any:
        assert self._cur_node is not None
        converter_packet = (
            DYNAMO_CONVERTERS.get_unvalidated(type(target))
            if isinstance(target, Plugin)
            else DYNAMO_CONVERTERS.get(self._cur_node)
        )
        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of function {torch.typename(target)} not currently supported!"
            )

        converter, calling_convention = converter_packet

        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def get_attr(self, target: str, args: Any, kwargs: Any) -> trt.ITensor:
        if target in self._constant_cache:
            return self._constant_cache[target][0]
        with _disable_current_modes():
            assert isinstance(
                constant_tensor := self.fetch_attr(target), torch.Tensor
            ), f"The fetched attribute '{target}' is not a PyTorch tensor: {constant_tensor}"
        constant_tensor = constant_tensor.cpu().detach().contiguous()
        trt_weight = trt.Weights(
            DataType(constant_tensor.dtype).to(trt.DataType),
            constant_tensor.data_ptr(),
            constant_tensor.numel(),
        )
        constant = self.ctx.net.add_constant(trt.Dims(list(constant_tensor.shape)), trt_weight)
        constant.name = target
        self._constant_cache[target] = (output := constant.get_output(0), constant_tensor)
        return output

    def output(self, target: str, args: Any, kwargs: Any) -> list[Any]:
        outputs = super().output(target, args, kwargs)
        if self.user_output_names is not None:
            for i, output_name in enumerate(self.user_output_names):
                if i >= len(outputs):
                    self.logger.warning(
                        f"The model has {len(outputs)} outputs, but got {len(self.user_output_names)} output names."
                    )
                    break
                if isinstance(outputs[i], trt.ITensor):
                    self.logger.debug(f"The {i}-th output will be renamed: {outputs[i].name} -> {output_name}")
                    outputs[i].name = output_name
                    self._output_names[i] = output_name
                else:
                    self.logger.warning(
                        f"The {i}-th output is not a ITensor object: {outputs[i]}. "
                        f"The given output name {output_name} will be discarded"
                    )
        return outputs


def _format_output(output: Any) -> str:
    """Format TensorRT output objects as strings.

    Args:
        output (Any): The output object to format

    Returns:
        str: A string representation of the output object
    """
    if isinstance(output, trt.ITensor):
        return (
            f"trt.ITensor(name={output.name}, shape={output.shape}, "
            f"dtype={output.dtype.name}, location={output.location.name})"
        )
    if isinstance(output, tuple):
        return f"({','.join(_format_output(x) for x in output)})"
    if isinstance(output, list):
        return f"[{','.join(_format_output(x) for x in output)}]"
    if isinstance(output, dict):
        tokens = (f"{key}: {_format_output(x)}" for key, x in output.items())
        return f"[{','.join(tokens)}]"
    return f"{type(output).__name__}({output})"


def infer_module_output_dtypes(
    module: GraphModule,
    truncate_double: bool = False,
) -> tuple[dtype, ...]:
    """Infer output data types from a graph module.

    Args:
        module (GraphModule): The graph module to analyze
        truncate_double (bool, optional): Whether to convert float64 to float32. Defaults to False.

    Returns:
        tuple[dtype, ...]: Tuple of inferred output data types

    Raises:
        RuntimeError: If a graph output node is missing tensor metadata
    """
    the_output = [n for n in module.graph.nodes if n.op == "output"][0]
    output_dtypes: list[dtype] = []
    for node in the_output.all_input_nodes:
        if not (tensor_meta := get_tensor_metadata(node)):
            raise RuntimeError(f"Found a graph output without tensor metadata: {node.format_node()}")
        output_dtype = tensor_meta.dtype
        output_dtypes.append(
            dtype.float32 if truncate_double and output_dtype == torch.float64 else dtype._from(output_dtype)
        )
    return tuple(output_dtypes)
