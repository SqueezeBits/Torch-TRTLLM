# pylint: disable=no-member
import logging
import re
from typing import Any

import tensorrt as trt
import torch
from torch.fx import GraphModule
from torch.fx.node import Node, Target
from torch_tensorrt import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import (
    DYNAMO_CONVERTERS,
    CallingConvention,
    TRTInterpreter,
    UnsupportedOperatorException,
)

from .fake_gpt_attention_plugin import FakeGPTAttentionPlugin


class DynamicTRTInterpreter(TRTInterpreter):
    def __init__(
        self,
        module: GraphModule,
        input_specs: tuple[Input, ...],
        logger_level: trt.ILogger.Severity = trt.ILogger.Severity.WARNING,
        output_dtypes: tuple[dtype, ...] | None = None,
        compilation_settings: CompilationSettings | None = None,
        engine_cache: BaseEngineCache | None = None,
        network_name: str | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            module,
            input_specs=input_specs,
            logger_level=logger_level,
            output_dtypes=output_dtypes,
            compilation_settings=compilation_settings or CompilationSettings(),
            engine_cache=engine_cache,
        )
        level = {
            trt.ILogger.Severity.ERROR: logging.ERROR,
            trt.ILogger.Severity.INTERNAL_ERROR: logging.ERROR,
            trt.ILogger.Severity.WARNING: logging.WARNING,
            trt.ILogger.Severity.INFO: logging.INFO,
            trt.ILogger.Severity.VERBOSE: logging.DEBUG,
        }.get(logger_level, logging.WARNING)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("TRTInterpreter:[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.placeholder_names = [n.name for n in module.graph.nodes if n.op == "placeholder"]
        self.user_output_names = output_names
        if network_name:
            self.ctx.net.name = network_name

    def _construct_trt_network_def(self) -> None:
        super()._construct_trt_network_def()

        def reformat_unnamed_layer(input_string: str) -> str:
            """E.g. "(Unnamed Layer* 1095) [Slice]" -> "slice_1095"."""
            # Use regex to capture the number and the part after the bracket
            match = re.search(r"\* (\d+)\) \[(\w+)\]", input_string)
            if match:
                # Extract the number and the word in brackets
                number = match.group(1)
                word = match.group(2).lower()  # Convert the word to lowercase
                # Return the formatted string
                return f"{word}_{number}"
            return input_string[:]

        def get_alias(tensor: trt.ITensor | None) -> str:
            if tensor is None:
                return "None"

            def _simplify(name: str) -> str:
                if (index := name.find("]_output")) != -1:
                    name = name[: index + 1]
                name = name.split("-")[-1]
                if name.startswith("[") and name.endswith("]"):
                    name = name.removeprefix("[").removesuffix("]")
                name = reformat_unnamed_layer(name)
                return name

            long_name = tensor.name
            while (short_name := _simplify(long_name)) != long_name:
                long_name = short_name
            return short_name

        def get_tensor_repr(tensor: trt.ITensor | None) -> str:
            if tensor is None:
                return "None"
            name = get_alias(tensor)
            dtype_ = tensor.dtype.name
            shape = tensor.shape
            device = tensor.location.name
            return f"{name}: {dtype_}{shape}@{device}"

        def get_network_ir(network: trt.INetworkDefinition) -> str:
            lines: list[str] = []
            for i in range(network.num_inputs):
                lines.append(f"{get_tensor_repr(network.get_input(i))} [network input]")
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                inputs = ", ".join(get_alias(layer.get_input(j)) for j in range(layer.num_inputs))
                outputs = ", ".join(get_tensor_repr(layer.get_output(j)) for j in range(layer.num_outputs))
                lines.append(f"{outputs} = {layer.type.name}({inputs})")
            for i in range(network.num_outputs):
                lines.append(f"{get_tensor_repr(network.get_output(i))} [network output]")
            return "\n".join(lines)

        with open(f"{self.ctx.net.name}.txt", "w") as f:
            f.write(get_network_ir(self.ctx.net))
        self.logger.info(f"TensorRT Network saved at {self.ctx.net.name}.txt")

    def run_node(self, n: Node) -> Node:
        self.logger.info(f"Converting {n.format_node(self.placeholder_names) or str(n)}")
        output = super().run_node(n)
        self.logger.info(f"{n.name} -> {_format_output(output)}")
        return output

    def call_function(self, target: Target, args: Any, kwargs: Any) -> Any:
        # TODO: Why is this stateful? We should be able to take in the inputs
        converter_packet = (
            DYNAMO_CONVERTERS.get_unvalidated(FakeGPTAttentionPlugin)
            if isinstance(target, FakeGPTAttentionPlugin)
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
                    outputs[i].name = output_name
                    self._output_names[i] = output_name
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
