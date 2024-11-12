from functools import cached_property

import numpy as np

# pylint: disable-next=unused-import
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
from tensorrt import ICudaEngine
from typing_extensions import Self

from ..types import StrictlyTyped


class Allocation(StrictlyTyped):
    host: np.ndarray
    device: cuda.DeviceAllocation
    shape: tuple[int, ...]

    @classmethod
    def allocate_for(cls, tensor: torch.Tensor) -> Self:
        dtype = tensor.numpy(force=True).dtype
        host = cuda.pagelocked_empty(tensor.numel(), dtype)
        return cls(
            host=host,
            device=cuda.mem_alloc(host.nbytes),
            shape=(*tensor.shape,),
        )

    @property
    def binding(self) -> int:
        return int(self.device)

    def get_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.host).reshape(self.shape)

    def is_compatible_with(self, tensor: torch.Tensor) -> bool:
        dtype = tensor.new_zeros(()).numpy(force=True).dtype
        return self.host.dtype == dtype and self.host.size == tensor.numel()


class TensorRTInferenceSession:
    def __init__(self, engine: trt.ICudaEngine | str) -> None:
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.engine = engine if isinstance(engine, trt.ICudaEngine) else self.load_engine(engine)
        self.context = self.engine.create_execution_context()
        self.inputs: dict[str, Allocation] = {}
        self.outputs: dict[str, Allocation] = {}

    @cached_property
    def stream(self) -> "cuda.Stream":
        return cuda.Stream()

    def load_engine(self, engine_path: str) -> ICudaEngine:
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    @cached_property
    def input_tensor_names(self) -> list[str]:
        return [
            tensor_name
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(tensor_name := self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]

    @cached_property
    def output_tensor_names(self) -> list[str]:
        return [
            tensor_name
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(tensor_name := self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

    def allocate_inputs(self, input_tensors: dict[str, torch.Tensor]) -> None:
        self._allocate(input_tensors, self.inputs)

    def allocate_outputs(self, output_tensors: dict[str, torch.Tensor]) -> None:
        self._allocate(output_tensors, self.outputs)

    def _allocate(self, tensors: dict[str, torch.Tensor], buffers: dict[str, Allocation]) -> None:
        assert buffers is self.inputs or buffers is self.outputs
        is_input = buffers is self.inputs
        tag = "input" if is_input else "output"
        expected_tensor_names = self.input_tensor_names if is_input else self.output_tensor_names
        assert expected_tensor_names == list(tensors), (
            f"Expected {tag} names to be {', '.join(expected_tensor_names)}, " f"but got {', '.join(tensors)}"
        )
        for name, tensor in tensors.items():
            if name in buffers and buffers[name].is_compatible_with(tensor):
                continue
            buffers[name] = (mem := Allocation.allocate_for(tensor))
            self.context.set_tensor_address(name, mem.binding)
            if is_input:
                self.context.set_input_shape(name, mem.shape)
            print(f"Allocated buffer for the {tag} {name}: {mem.shape} | {mem.host.dtype}")

    def run(self, input_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.allocate_inputs(input_tensors)
        device = next(iter(input_tensors.values())).device
        for i, (name, tensor) in enumerate(input_tensors.items()):
            assert name == self.engine.get_tensor_name(i)
            data = tensor.numpy(force=True)
            input_memory = self.inputs[name]
            np.copyto(input_memory.host, data.ravel())
            cuda.memcpy_htod_async(input_memory.device, input_memory.host, self.stream)
            self.context.set_tensor_address(name, input_memory.binding)

        for name, output_memory in self.outputs.items():
            self.context.set_tensor_address(name, output_memory.binding)

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()

        for output_memory in self.outputs.values():
            cuda.memcpy_dtoh_async(output_memory.host, output_memory.device, self.stream)

        return {name: output_memory.get_tensor().to(device) for name, output_memory in self.outputs.items()}
