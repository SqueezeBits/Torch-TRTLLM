import json
from typing import Any

import tensorrt as trt
import torch
import torch.fx
from loguru import logger
from torch.fx import GraphModule
from torch.fx.node import Node, Target
from torch.fx.passes.shape_prop import TensorMetadata
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
from .fx.targets import GemmPlugin, GPTAttentionPlugin
from .types import DataType


class TRTLLMInterpreter(TRTInterpreter):
    def __init__(
        self,
        module: GraphModule,
        input_specs: tuple[Input, ...],
        network_flags: TensorRTNetworkCreationFlags,
        builder_config: TensorRTBuilderConfig,
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
        if network_name:
            self.ctx.net.name = network_name
        self._constant_cache: dict[str, trt.ITensor] = {}
        self._constant_tensors: list[trt.ITensor] = []

    def validate_conversion(self) -> set[str]:
        missing_ops = super().validate_conversion()
        return {missing_op for missing_op in missing_ops if "gpt_attention_plugin" not in missing_op}

    def _construct_trt_network_def(self) -> None:
        super()._construct_trt_network_def()
        save_for_debug("trt_network_def", self.ctx.net, self.optimization_profiles)

    def _populate_trt_builder_config(
        self,
        strict_type_constraints: bool = False,
        algorithm_selector: trt.IAlgorithmSelector | None = None,
        tactic_sources: int | None = None,
    ) -> trt.IBuilderConfig:
        builder_config = self.builder.create_builder_config()
        self._builder_config.copy_to(builder_config)
        with open_debug_artifact("builder_config.json") as f:
            if f:
                json.dump(builder_config_as_dict(builder_config), f, indent=2, sort_keys=True)
        return builder_config

    def run_node(self, n: Node) -> Node:
        self.logger.debug(f"Converting {n.format_node(self.placeholder_names) or str(n)}")
        output = super().run_node(n)
        self.logger.debug(f"{n.name} -> {_format_output(output)}")
        return output

    def call_function(self, target: Target, args: Any, kwargs: Any) -> Any:
        assert self._cur_node is not None
        converter_packet = (
            DYNAMO_CONVERTERS.get_unvalidated(type(target))
            if isinstance(target, GPTAttentionPlugin | GemmPlugin)
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
            return self._constant_cache[target]
        with _disable_current_modes():
            frozen_attr = self.fetch_attr(target)
            if isinstance(frozen_attr, torch.nn.Parameter):
                constant_tensor = frozen_attr.data
            else:
                constant_tensor = frozen_attr
        constant_tensor = constant_tensor.cpu().detach().contiguous()
        trt_weight = trt.Weights(
            DataType(constant_tensor.dtype).to(trt.DataType),
            constant_tensor.data_ptr(),
            torch.numel(constant_tensor),
        )
        constant = self.ctx.net.add_constant(constant_tensor.shape, trt_weight)
        constant.name = target
        self._constant_tensors.append(constant_tensor)
        self._constant_cache[target] = constant.get_output(0)
        return constant.get_output(0)

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
    the_output = [n for n in module.graph.nodes if n.op == "output"][0]
    output_dtypes: list[dtype] = []
    for node in the_output.all_input_nodes:
        if not isinstance((tensor_meta := node.meta.get("tensor_meta", None)), TensorMetadata):
            raise RuntimeError(f"Found a graph output without tensor metadata: {node.format_node()}")
        output_dtype = tensor_meta.dtype
        output_dtypes.append(
            dtype.float32 if truncate_double and output_dtype == torch.float64 else dtype._from(output_dtype)
        )
    return tuple(output_dtypes)
