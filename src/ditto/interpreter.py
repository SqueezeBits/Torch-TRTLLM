# pylint: disable=no-member
import json
import os
from pathlib import Path
from typing import Any

import tensorrt as trt
import tensorrt_llm as trtllm
import torch
from loguru import logger
from torch.fx import GraphModule
from torch.fx.node import Node, Target
from torch_tensorrt import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion import (
    DYNAMO_CONVERTERS,
    CallingConvention,
    ConversionContext,
    TRTInterpreter,
    UnsupportedOperatorException,
)
from torch_tensorrt.dynamo.conversion.converter_utils import create_constant

from .config import DEFAULT_TRT_PROFILING_VERBOSITY
from .debug import (
    EngineInfo,
    builder_config_as_dict,
    get_dynamic_input_ranges,
    get_human_readable_flags,
    open_debug_artifact,
    save_onnx_without_weights,
)
from .fake_targets import FakeGPTAttentionPlugin


class TRTLLMInterpreter(TRTInterpreter):
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
        strongly_typed: bool = True,
    ) -> None:
        super().__init__(
            module,
            input_specs=input_specs,
            logger_level=logger_level,
            output_dtypes=output_dtypes,
            compilation_settings=compilation_settings or CompilationSettings(),
            engine_cache=engine_cache,
        )

        flags = 1 << trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.value
        if strongly_typed:
            flags |= 1 << trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED.value

        self.ctx = ConversionContext(self.builder.create_network(flags), self.compilation_settings)
        self.logger = logger
        self.placeholder_names = [n.name for n in module.graph.nodes if n.op == "placeholder"]
        self.user_output_names = output_names
        if network_name:
            self.ctx.net.name = network_name
        self._constant_cache: dict[str, trt.ITensor] = {}

    def _construct_trt_network_def(self) -> None:
        super()._construct_trt_network_def()
        with open_debug_artifact("trt_network_def.onnx", "wb") as f:
            if f:
                shape_ranges = (
                    get_dynamic_input_ranges(self.ctx.net, profiles)
                    if (profiles := self.optimization_profiles)
                    else None
                )
                flags = get_human_readable_flags(self.ctx.net)
                proto = EngineInfo.from_network_definition(self.ctx.net).as_onnx(shape_ranges, flags)
                save_onnx_without_weights(proto, f)

    def _populate_trt_builder_config(
        self,
        strict_type_constraints: bool = False,
        algorithm_selector: trt.IAlgorithmSelector | None = None,
        tactic_sources: int | None = None,
        timing_cache: str | Path | trt.ITimingCache | None = None,
        use_refit: bool = False,
        profiling_verbosity: trt.ProfilingVerbosity = DEFAULT_TRT_PROFILING_VERBOSITY,
        use_strip_plan: bool = False,
        weight_streaming: bool = False,
        weight_sparsity: bool = False,
    ) -> trt.IBuilderConfig:
        assert (timing_cache is None) or (
            algorithm_selector is None
        ), "timing cache and algorithm selector cannot be specified at the same time"

        builder_config = super()._populate_trt_builder_config(
            # Workaround for the bug mentioned in https://github.com/pytorch/TensorRT/pull/3300
            strict_type_constraints=False,
            algorithm_selector=algorithm_selector,
            tactic_sources=tactic_sources,
        )

        if strict_type_constraints:
            builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        if weight_streaming:
            builder_config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

        if use_refit:
            builder_config.set_flag(trt.BuilderFlag.REFIT)

        # Use fine-grained refit when strip plan is enabled in TRT10.2+.
        if use_strip_plan:
            builder_config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)

        if use_strip_plan:
            builder_config.set_flag(trt.BuilderFlag.STRIP_PLAN)

        # Set TRT Engine profiling verbosity
        builder_config.profiling_verbosity = profiling_verbosity

        # set timing cache
        cache = None
        if timing_cache is not None:
            # use given cache
            if isinstance(timing_cache, trt.ITimingCache):
                cache = timing_cache
            # read cache from file
            elif isinstance(timing_cache, str | Path) and os.path.exists(timing_cache):
                with open(timing_cache, "rb") as f:
                    cache = builder_config.create_timing_cache(f.read())
            else:
                trtllm.logger.warning("Invalid timing cache, using freshly created one")
        if cache is None:
            cache = builder_config.create_timing_cache(b"")
        # When user does not given any existing cache, internally always created one
        # so the cache should never None here
        assert cache is not None and isinstance(cache, trt.ITimingCache)
        builder_config.set_timing_cache(cache, ignore_mismatch=False)

        # set weight sparsity
        if weight_sparsity:
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

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

    def get_attr(self, target: str, args: Any, kwargs: Any) -> trt.ITensor:
        if target in self._constant_cache:
            return self._constant_cache[target]
        numpy_array = super().get_attr(target, args, kwargs)
        constant = create_constant(self.ctx, numpy_array, target, numpy_array.dtype)
        self._constant_cache[target] = constant
        return constant

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
                    self.logger.info(f"The {i}-th output will be renamed: {outputs[i].name} -> {output_name}")
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
