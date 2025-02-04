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

import tensorrt as trt
from loguru import logger
from tensorrt_llm._common import _is_building
from torch.fx import GraphModule
from torch_tensorrt._Device import Device
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException
from torch_tensorrt.logging import TRT_LOGGER

from .arguments import TRTLLMArgumentHint
from .configs import TensorRTConfig
from .debug import save_for_debug, should_save_debug_artifacts
from .interpreter import TRTLLMInterpreter

CURRENT_DEVICE = Device._current_device()


@_is_building  # type: ignore
def convert(
    graph_module: GraphModule,
    argument_hint: TRTLLMArgumentHint,
    trt_config: TensorRTConfig,
    rank: int,
    *,
    engine_cache: BaseEngineCache | None = None,
    network_name: str | None = None,
    output_names: list[str] | None = None,
) -> bytes:
    """Convert an graph module to a TensorRT engine."""
    input_specs = tuple(
        tensor_type_hint.as_spec(name)
        for name, tensor_type_hint in argument_hint.as_dict().items()
        if tensor_type_hint is not None
    )
    logger.opt(lazy=True).debug("input_specs:\n{x}", x=lambda: "\n".join(str(spec) for spec in input_specs))

    assert len(placeholders := graph_module.graph.find_nodes(op="placeholder")) == len(input_specs)
    assert all(p.name == i.name for p, i in zip(placeholders, input_specs))

    try:
        interpreter = TRTLLMInterpreter(
            graph_module,
            input_specs,
            builder_config=trt_config.builder_config,
            network_flags=trt_config.network_creation_flags,
            rank=rank,
            engine_cache=engine_cache,
            network_name=network_name,
            output_names=output_names,
        )
        engine = interpreter.run().serialized_engine
        if should_save_debug_artifacts():
            save_for_debug(
                f"trt_engine_rank{rank}",
                trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine),
            )
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
