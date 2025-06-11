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

import torch
from loguru import logger
from pydantic import model_validator
from torch.fx import GraphModule
from typing_extensions import Self

from ...fx.utils import get_val
from ...types import DataType, DTypeLiteral, StrictlyTyped
from ..tensorrt import TensorRTConfig
from .model import TRTLLMModelConfig
from .optimization_profile import RuntimeTRTLLMOptimizationProfileConfig, TRTLLMOptimizationProfileConfig


class TRTLLMBuildConfig(RuntimeTRTLLMOptimizationProfileConfig, TRTLLMModelConfig):
    """Minimal subset of properties in `trtllm.BuildConfig` required at runtime."""

    @classmethod
    def merge(
        cls,
        profile_config: TRTLLMOptimizationProfileConfig,
        model_config: TRTLLMModelConfig,
    ) -> Self:
        """Create a new instance by merging the profile and model configurations.

        Args:
            profile_config (TRTLLMOptimizationProfileConfig): The profile configuration.
            model_config (TRTLLMModelConfig): The model configuration.

        Returns:
            Self: The merged configuration.
        """
        return cls.model_validate(
            {
                **profile_config.runtime().model_dump(),
                **model_config.model_dump(),
            }
        )

    @model_validator(mode="after")
    def check_context_mha_dependencies(self) -> Self:
        """Check conditions imposed by `context_mha`.

        The criterions and error messages are adopted from `tensorrt_llm._common.check_max_num_tokens`.
        While TRT-LLM tries to adjust the wrong values provided by the user, we will simply reject them.

        Returns:
            Self: The validated instance.
        """
        assert self.plugin_config.context_fmha or self.max_num_tokens >= self.max_input_len, (
            f"When {self.plugin_config.context_fmha=}, {self.max_num_tokens=}) "
            f"should be at least {self.max_input_len=}."
        )
        assert not self.plugin_config.context_fmha or self.max_num_tokens >= self.plugin_config.tokens_per_block, (
            f"When {self.plugin_config.context_fmha=}, {self.max_num_tokens=} "
            f"should be at least {self.plugin_config.tokens_per_block=}."
        )
        return self


class TRTMultiModalBuildConfig(StrictlyTyped):
    """Configuration for building TensorRT engines for a multimodel model.

    Args:
        model_name (str): The name of the model.
        model_type (str): The type of the model.
        precision (DTypeLiteral): The precision of the model.
        strongly_typed (bool): Whether to use strongly typed mode.
        max_batch_size (int): The maximum batch size.
        tensor_parallel (int): The tensor parallel.
        output_shape (list[int]): The output shape.
    """

    model_name: str
    model_type: str
    precision: DTypeLiteral
    strongly_typed: bool
    max_batch_size: int
    tensor_parallel: int
    output_shape: list[int]

    @classmethod
    def create_from(
        cls,
        graph_module: GraphModule,
        trt_config: TensorRTConfig,
        *,
        model_name: str,
        model_type: str,
        dtype: torch.dtype,
        max_batch_size: int,
        tensor_parallel: int,
    ) -> Self:
        """Create a new instance by merging the profile and model configurations.

        Args:
            graph_module (GraphModule): The graph module to convert to TensorRT engines
            trt_config (TensorRTConfig): The TensorRT configuration.
            model_name (str): The name of the model.
            model_type (str): The type of the model.
            dtype (torch.dtype): The data type of the model.
            max_batch_size (int): The maximum batch size.
            tensor_parallel (int): The tensor parallel.

        Returns:
            Self: The merged configuration.
        """
        return cls(
            model_name=model_name,
            model_type=model_type,
            precision=DataType(dtype).to(DTypeLiteral),
            strongly_typed=trt_config.network_creation_flags.strongly_typed,
            max_batch_size=max_batch_size,
            tensor_parallel=tensor_parallel,
            output_shape=get_output_shape(graph_module),
        )

    def as_dict(self) -> dict[str, Any]:
        """Convert builder config to dictionary.

        Returns:
            dict[str, Any]: Dictionary of builder config
        """
        return {"builder_config": self.model_dump()}


def get_output_shape(graph_module: GraphModule) -> list[int]:
    """Get the output shape of the graph module.

    Args:
        graph_module (GraphModule): The graph module to get the output shape from.

    Returns:
        list[int]: The output shape of the graph module.
    """
    output_nodes = graph_module.graph.find_nodes(op="output")
    assert len(output_nodes) > 0, "No output node found"
    if len(output_nodes) > 1:
        logger.warning(f"Multiple output nodes found to found the output shape, using the first one: {output_nodes[0]}")
    if len(all_input_nodes := output_nodes[0].all_input_nodes) > 1:
        logger.warning(
            f"Multiple input nodes found to found the output shape, using the first one: {all_input_nodes[0]}"
        )
    assert isinstance(output_tensor := get_val(all_input_nodes[0]), torch.Tensor), ""

    return [s if isinstance(s, int) else -1 for s in output_tensor.shape]
