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

import numpy as np
import tensorrt as trt
import torch
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from torch.fx import Graph, Node
from typing_extensions import Self

from ...types import StrictlyTyped
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin


class AllReducePlugin(Plugin):
    """TensorRT plugin implementation of All Reduce.

    Attributes:
        group (list[int]): The group of ranks to reduce from
        type_id (trt.DataType): The data type of the tensor to reduce
        strategy (AllReduceStrategy): The strategy to use for all reduce
        fusion_op (AllReduceFusionOp): The fusion operation to use for all reduce
        eps (float): The epsilon value for all reduce
        affine (bool): Whether to apply affine transformation to the tensor
        bias (bool): Whether to apply bias to the tensor
        scale (bool): Whether to apply scale to the tensor
    """

    # TODO: eps, affine, bias, scale are not supported currently

    # the order of the attributes does matter!
    group: list[int]
    type_id: trt.DataType
    strategy: AllReduceStrategy = AllReduceStrategy.AUTO
    fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE
    eps: float = 1e-5
    affine: bool = False
    bias: bool = False
    scale: bool = False

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        if name in ("config", "fusion_op"):
            return np.int8
        return super().get_field_dtype(name, value)

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if is_in_fake_tensor_mode():
            return x
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")


class AllReducePluginInputs(StrictlyTyped):
    """Inputs for All Reduce plugin.

    Attributes:
        workspace (Node | None): Workspace node
    """

    workspace: Node | None

    @classmethod
    def find_from(cls, graph: Graph, allreduce_plugin: AllReducePlugin) -> Self:
        """Find the inputs for All Reduce plugin.

        Args:
            graph (Graph): The graph to find the inputs from
            allreduce_plugin (AllReducePlugin): The All Reduce plugin to find the inputs for

        Returns:
            Self: The inputs for All Reduce plugin
        """
        existing_placeholders = {p.name: p for p in graph.find_nodes(op="placeholder")}
        workspace = None
        if allreduce_plugin.strategy not in (AllReduceStrategy.NCCL, AllReduceStrategy.UB):
            workspace = existing_placeholders.get("all_reduce_workspace", None)
        if allreduce_plugin.fusion_op != AllReduceFusionOp.NONE:
            # TODO: add additional inputs for other fusion ops
            raise NotImplementedError("AllReduceFusionOp is not supported yet")

        return cls(workspace=workspace)
