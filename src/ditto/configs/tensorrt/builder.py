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
    DLA_core: int = -1
    algorithm_selector: trt.IAlgorithmSelector | None = None
    avg_timing_iterations: int = 1
    builder_optimization_level: int = Field(default=3, ge=1, le=5)
    default_device_type: trt.DeviceType = trt.DeviceType.GPU
    engine_capability: trt.EngineCapability = trt.EngineCapability.STANDARD
    flags: TensorRTBuilderFlags = Field(default_factory=TensorRTBuilderFlags)
    hardware_compatibility_level: trt.HardwareCompatibilityLevel = trt.HardwareCompatibilityLevel.NONE
    int8_calibrator: trt.IInt8Calibrator | None = None
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
        for name, value in self.model_dump().items():
            if hasattr(native_config, name) and getattr(native_config, name) != value:
                logger.debug(f"Setting '{name}' of trt.IBuilderConfig to {value}")
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
        if self.profiling_verbosity == trt.ProfilingVerbosity.DETAILED and self.progress_monitor is None:
            logger.info(
                f"Automatically setting progress monitor to {BuildMonitor.__name__} "
                "as the profiling verbosity is 'DETAILED'"
            )
            self.progress_monitor = BuildMonitor()
        return self

    @field_serializer("flags", "quantization_flags", return_type=int)
    def serialize_bitmask_flags(self, flags: BitmaskFlags) -> int:
        return flags.bitmask

    @field_validator("flags", mode="before")
    @classmethod
    def validate_flags(cls, flags: Any) -> Any:
        if isinstance(flags, int):
            return TensorRTBuilderFlags.from_bitmask(flags)
        return flags

    @field_validator("quantization_flags", mode="before")
    @classmethod
    def validate_quantization_flags(cls, flags: Any) -> Any:
        if isinstance(flags, int):
            return TensorRTQuantizationFlags.from_bitmask(flags)
        return flags
