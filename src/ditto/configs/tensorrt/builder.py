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

from typing import Any

import tensorrt as trt
from loguru import logger
from pydantic import Field, field_serializer, field_validator, model_validator
from torch_tensorrt import Device, DeviceType, EngineCapability
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo._defaults import DLA_GLOBAL_DRAM_SIZE, DLA_LOCAL_DRAM_SIZE, DLA_SRAM_SIZE, default_device

try:
    from torch_tensorrt.dynamo.conversion._TRTBuilderMonitor import _RichMonitor as BuildMonitor
except ImportError:
    from torch_tensorrt.dynamo.conversion._TRTBuilderMonitor import _ASCIIMonitor as BuildMonitor
from typing_extensions import Self

from ...constants import DEFAULT_TRT_PROFILING_VERBOSITY
from ...types import StrictlyTyped
from .flags import BitmaskFlags, TensorRTBuilderFlags, TensorRTQuantizationFlags


class TensorRTBuilderConfig(StrictlyTyped):
    """Configuration for TensorRT builder.

    Args:
        DLA_core (int): The DLA core to use. Defaults to -1.
        algorithm_selector (trt.IAlgorithmSelector | None): The algorithm selector to use. Defaults to None.
        avg_timing_iterations (int): The number of timing iterations to use. Defaults to 1.
        builder_optimization_level (int): The builder optimization level to use. Defaults to 3.
        default_device_type (trt.DeviceType): The default device type to use. Defaults to trt.DeviceType.GPU
        engine_capability (trt.EngineCapability): The engine capability to use.
            Defaults to trt.EngineCapability.STANDARD.
        flags (TensorRTBuilderFlags): The flags to use. Defaults to TensorRTBuilderFlags().
        hardware_compatibility_level (trt.HardwareCompatibilityLevel): The hardware compatibility level to use.
            Defaults to trt.HardwareCompatibilityLevel.NONE.
        int8_calibrator (trt.IInt8Calibrator | None): The int8 calibrator to use. Defaults to None.
        max_aux_streams (int): The maximum number of auxiliary streams to use. Defaults to -1.
        plugins_to_serialize (list[str]): The plugins to serialize. Defaults to an empty list.
        profile_stream (int): The profile stream to use. Defaults to 0.
        profiling_verbosity (trt.ProfilingVerbosity): The profiling verbosity to use.
            Defaults to DEFAULT_TRT_PROFILING_VERBOSITY.
        progress_monitor (trt.IProgressMonitor | None): The progress monitor to use. Defaults to None.
        quantization_flags (TensorRTQuantizationFlags): The quantization flags to use.
            Defaults to TensorRTQuantizationFlags().
        runtime_platform (trt.RuntimePlatform): The runtime platform to use.
            Defaults to trt.RuntimePlatform.SAME_AS_BUILD.
        memory_pool_limits (dict[trt.MemoryPoolType, int]): The memory pool limits to use. Defaults to an empty dict.
        optimization_profiles (list[trt.IOptimizationProfile]): The optimization profiles to use.
            Defaults to an empty list.
    """

    DLA_core: int = -1
    algorithm_selector: trt.IAlgorithmSelector | None = None
    avg_timing_iterations: int = 1
    builder_optimization_level: int = Field(default=3, ge=1, le=5)
    default_device_type: trt.DeviceType = trt.DeviceType.GPU
    engine_capability: trt.EngineCapability = trt.EngineCapability.STANDARD
    flags: TensorRTBuilderFlags = Field(default_factory=TensorRTBuilderFlags)
    hardware_compatibility_level: trt.HardwareCompatibilityLevel = trt.HardwareCompatibilityLevel.NONE
    int8_calibrator: trt.IInt8Calibrator | None = Field(default=None, deprecated=True)
    max_aux_streams: int = -1
    plugins_to_serialize: list[str] = Field(default_factory=list)
    profile_stream: int = 0
    profiling_verbosity: trt.ProfilingVerbosity = DEFAULT_TRT_PROFILING_VERBOSITY
    progress_monitor: trt.IProgressMonitor | None = None
    quantization_flags: TensorRTQuantizationFlags = Field(default_factory=TensorRTQuantizationFlags)
    runtime_platform: trt.RuntimePlatform = trt.RuntimePlatform.SAME_AS_BUILD
    memory_pool_limits: dict[trt.MemoryPoolType, int] = Field(default_factory=dict, exclude=True)
    optimization_profiles: list[trt.IOptimizationProfile] = Field(default_factory=list, exclude=True)

    def copy_to(self, native_config: trt.IBuilderConfig) -> None:
        """Copy the configuration to a native TensorRT builder configuration.

        Args:
            native_config (trt.IBuilderConfig): The native TensorRT builder configuration to copy the configuration to.
        """
        for name, value in self.model_dump().items():
            # Skip deprecated option `int8_calibrator` if its value is None
            # Otherwise, the TensorRT will let the user know the option is deprecated
            if name == "int8_calibrator" and value is None:
                continue
            if hasattr(native_config, name) and getattr(native_config, name) != value:
                logger.debug(f"Setting attribute '{name}' of trt.IBuilderConfig to {value}")
                setattr(native_config, name, value)
        for pool, pool_size in self.memory_pool_limits.items():
            logger.debug(f"Setting memory limit of '{pool}' to {pool_size}")
            native_config.set_memory_pool_limit(pool, pool_size)
        # pylint: disable-next=not-an-iterable
        for profile in self.optimization_profiles:
            if native_config.add_optimization_profile(profile) == -1:
                logger.warning(f"Failed to add optimization profile: {profile}")

    def get_compilation_settings(
        self,
        *,
        device_config: Device | None = None,
    ) -> CompilationSettings:
        """Create compilation settings for the TensorRT builder.

        Args:
            device_config (Device | None): The device configuration to use. Defaults to None.

        Returns:
            CompilationSettings: Compilation settings for the TensorRT builder.
        """
        device_config = device_config or default_device()
        device_config.dla_core = self.DLA_core
        if self.DLA_core > 0:
            device_config.device_type = DeviceType.DLA
        return CompilationSettings(
            enabled_precisions=self.flags.enabled_precisions,
            debug=self.profiling_verbosity == trt.ProfilingVerbosity.DETAILED,
            workspace_size=self.memory_pool_limits.get(trt.MemoryPoolType.WORKSPACE, 0),
            max_aux_streams=self.max_aux_streams,
            version_compatible=self.flags.version_compatible and self.flags.exclude_lean_runtime,
            hardware_compatible=self.hardware_compatibility_level == trt.HardwareCompatibilityLevel.AMPERE_PLUS,
            optimization_level=self.builder_optimization_level,
            engine_capability=EngineCapability._from(self.engine_capability),
            num_avg_timing_iters=self.avg_timing_iterations,
            device=device_config,
            dla_global_dram_size=self.memory_pool_limits.get(trt.MemoryPoolType.DLA_GLOBAL_DRAM, DLA_GLOBAL_DRAM_SIZE),
            dla_local_dram_size=self.memory_pool_limits.get(trt.MemoryPoolType.DLA_LOCAL_DRAM, DLA_LOCAL_DRAM_SIZE),
            dla_sram_size=self.memory_pool_limits.get(trt.MemoryPoolType.DLA_MANAGED_SRAM, DLA_SRAM_SIZE),
            sparse_weights=self.flags.sparse_weights,
            disable_tf32=not self.flags.tf32,
            make_refittable=self.flags.refit,
        )

    @model_validator(mode="after")
    def adjust_progress_monitor(self) -> Self:
        """Set the progress monitor if the profiling verbosity is 'DETAILED' after the model instantiation.

        Returns:
            Self: The validated instance.
        """
        if self.profiling_verbosity == trt.ProfilingVerbosity.DETAILED and self.progress_monitor is None:
            logger.info(
                f"Automatically setting progress monitor to {BuildMonitor.__name__.removeprefix('_')} "
                "as the profiling verbosity is 'DETAILED'"
            )
            self.progress_monitor = BuildMonitor()
        return self

    @field_serializer("flags", "quantization_flags", return_type=int)
    def serialize_bitmask_flags(self, flags: BitmaskFlags) -> int:
        """Serialize the bitmask flags to an integer.

        Args:
            flags (BitmaskFlags): The bitmask flags to serialize.

        Returns:
            int: The serialized bitmask flags.
        """
        return flags.bitmask

    @field_validator("flags", mode="before")
    @classmethod
    def validate_flags(cls, flags: Any) -> Any:
        """Validate the flags.

        Args:
            flags (Any): The flags to validate.

        Returns:
            Any: The validated flags.
        """
        if isinstance(flags, int):
            return TensorRTBuilderFlags.from_bitmask(flags)
        return flags

    @field_validator("quantization_flags", mode="before")
    @classmethod
    def validate_quantization_flags(cls, flags: Any) -> Any:
        """Validate the quantization flags.

        Args:
            flags (Any): The quantization flags to validate.

        Returns:
            Any: The validated quantization flags.
        """
        if isinstance(flags, int):
            return TensorRTQuantizationFlags.from_bitmask(flags)
        return flags
