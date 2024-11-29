from __future__ import annotations

import tensorrt as trt
from loguru import logger
from tensorrt_llm._common import _is_building
from torch.fx import GraphModule
from torch_tensorrt._Device import Device
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException
from torch_tensorrt.logging import TRT_LOGGER

from ditto.configs.tensorrt.config import TensorRTConfig

from .arguments import TRTLLMArgumentHint
from .debug import save_for_debug
from .interpreter import TRTLLMInterpreter

CURRENT_DEVICE = Device._current_device()


@_is_building  # type: ignore
def convert(
    graph_module: GraphModule,
    argument_hint: TRTLLMArgumentHint,
    trt_config: TensorRTConfig,
    *,
    engine_cache: BaseEngineCache | None = None,
    network_name: str | None = None,
    output_names: list[str] | None = None,
) -> trt.ICudaEngine:
    """Convert an graph module to a TensorRT engine."""
    input_specs = tuple(tensor_type_hint.as_spec(name) for name, tensor_type_hint in argument_hint.as_dict().items())
    logger.opt(lazy=True).debug("input_specs:\n{x}", x=lambda: "\n".join(str(spec) for spec in input_specs))

    try:
        interpreter = TRTLLMInterpreter(
            graph_module,
            input_specs,
            builder_config=trt_config.builder_config,
            network_flags=trt_config.network_creation_flags,
            engine_cache=engine_cache,
            network_name=network_name,
            output_names=output_names,
        )
        result = interpreter.run()
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(result.serialized_engine)
        save_for_debug("trt_engine", engine)
        return engine
    except UnsupportedOperatorException as e:
        logger.error(
            f"Conversion of module {graph_module} not currently fully supported or convertible!",
            exc_info=True,
        )
        raise e
    except Exception as e:
        logger.error(
            f"While interpreting the module got an error: {e}",
            exc_info=True,
        )
        raise e
