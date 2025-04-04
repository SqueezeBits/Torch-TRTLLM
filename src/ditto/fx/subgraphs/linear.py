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

from typing import Literal

import torch
from loguru import logger
from torch._subclasses import FakeTensor
from torch.fx import Node
from typing_extensions import Self

from ...literals import ExpertTypeLiteral, LoraPluginInputPrefix
from ...types import verify
from ..metadata_keys import (
    ACTIVATION_QUANTIZATION,
    EXPERT_TYPE,
    FREE_LORA_PROTO,
    LAYER_INDEX,
    LORA_PREFIX,
    LORA_PROTOS,
)
from ..nodes import (
    MM,
    AddTensorTensor,
    Dequantize,
    Fp8RowwiseGemm,
    Gemm,
    Reshape,
    WeightOnlyGroupwiseQuantMatmul,
    WeightOnlyQuantMatmul,
)
from ..targets import ActivationQuantization, LoraProto
from ..utils import get_val
from .subgraph import Subgraph

MMType = MM | Fp8RowwiseGemm | Gemm | WeightOnlyGroupwiseQuantMatmul | WeightOnlyQuantMatmul


# pylint: disable-next=too-many-public-methods
class Linear(Subgraph):
    """A subgraph representing a linear layer.

    This subgraph identifies a pattern of matrix multiplication with an optional bias addition,
    which is equivalent to a linear/dense layer in neural networks.
    The matrix multiplication node can be either a MM or a Gemm node.

    The layer performs: output = input @ weight.T + bias

    Attributes:
        mm (MM | Gemm): The matrix multiplication operation node
        add (AddTensor | None): The bias addition operation node, if present
    """

    mm: MMType
    add: AddTensorTensor | None

    @property
    def weight_node(self) -> Node:
        """The weight parameter node."""
        return self.mm.other

    @property
    def weight_tensor(self) -> FakeTensor:
        """The weight parameter tensor."""
        assert (weight := get_val(self.mm.other, FakeTensor)) is not None
        return weight

    @property
    def has_transposed_weight(self) -> bool:
        """Whether the weight is transposed."""
        if isinstance(self.mm, Gemm):
            return self.mm.target.transb == 1
        return isinstance(self.mm, Fp8RowwiseGemm)

    @property
    def weight_in_features_dim(self) -> Literal[0, 1]:
        """The dimension representing the input features in the weight tensor."""
        return 1 if self.has_transposed_weight else 0

    @property
    def weight_out_features_dim(self) -> Literal[0, 1]:
        """The dimension representing the output features in the weight tensor."""
        return 0 if self.has_transposed_weight else 1

    @property
    def in_features(self) -> int:
        """The number of input features to the linear layer."""
        return self.weight_tensor.shape[self.weight_in_features_dim]

    @property
    def out_features(self) -> int:
        """The number of output features from the linear layer."""
        return self.weight_tensor.shape[self.weight_out_features_dim]

    @property
    def dtype(self) -> torch.dtype:
        """The output data type of the linear layer."""
        return self.weight_tensor.dtype

    @property
    def bias_node(self) -> Node | None:
        """The bias parameter node if present."""
        return self.add.other if self.add is not None else None

    @property
    def bias_tensor(self) -> FakeTensor | None:
        """The bias parameter tensor if present."""
        if self.add is not None:
            return get_val(self.add.other, FakeTensor)
        return None

    @property
    def has_transposed_input(self) -> bool:
        """Whether the input is transposed."""
        if isinstance(self.mm, Gemm):
            return self.mm.target.transa == 1
        return False

    @property
    def input_feature_dim(self) -> Literal[0, 1]:
        """The dimension representing the input features in the input tensor."""
        return 0 if self.has_transposed_input else 1

    @property
    def input_node(self) -> Node:
        """The input tensor node to the linear layer."""
        return self.mm.this

    @property
    def output_node(self) -> Node:
        """The output tensor node, either the bias addition or matrix multiplication result."""
        return self.add.node if self.add is not None else self.mm.node

    @property
    def reshape_in(self) -> Reshape | None:
        """The reshape operation before the linear layer if present."""
        return Reshape.specialize_from(self.mm.this)

    @property
    def reshape_out(self) -> Reshape | None:
        """The reshape operation after the linear layer if present."""
        if len(users := list(self.output_node.users)) != 1:
            return None
        return Reshape.specialize_from(users[0])

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (
                mm := MM.specialize_from(node)
                or Fp8RowwiseGemm.specialize_from(node)
                or Gemm.specialize_from(node)
                or WeightOnlyGroupwiseQuantMatmul.specialize_from(node)
                or WeightOnlyQuantMatmul.specialize_from(node)
            )
            and (input_node := get_val(mm.this, torch.Tensor)) is not None
            and (weight := get_val(mm.other, torch.Tensor)) is not None
            and input_node.ndim == 2
            and weight.ndim == 2
        ):
            return None

        add = AddTensorTensor.specialize_from(users[0]) if len(users := list(mm.users)) == 1 else None
        has_transposed_weight = isinstance(mm, Gemm) and mm.target.transb == 1
        if add is not None and not (
            add.this == mm.node
            and (bias := get_val(add.other, torch.Tensor)) is not None
            and bias.ndim == 1
            and bias.shape[-1] == weight.shape[0 if has_transposed_weight else 1]
        ):
            add = None
        return cls(mm=mm, add=add)

    @property
    def free_lora_proto(self) -> LoraProto | None:
        """The free LoRA prototype associated with this linear layer."""
        return verify(self.mm.meta.get(FREE_LORA_PROTO, None), as_type=LoraProto)

    @free_lora_proto.setter
    def free_lora_proto(self, value: LoraProto) -> None:
        """Set the free LoRA prototype for this linear layer."""
        assert FREE_LORA_PROTO not in self.mm.meta, f"Free lora proto already set for {self.mm}"
        self.mm.meta[FREE_LORA_PROTO] = value

    def bind_free_lora_proto(self, *, with_prefix: LoraPluginInputPrefix) -> None:
        """Bind a free LoRA prototype to this linear layer with the given prefix.

        Args:
            with_prefix: Prefix to bind the LoRA prototype with
        """
        self.mm.meta[LORA_PREFIX] = with_prefix
        if (lora_proto := verify(self.mm.meta.pop(FREE_LORA_PROTO, None), as_type=LoraProto)) is None:
            return
        assert LORA_PROTOS not in self.mm.meta, f"Lora protos already set for {self.mm}: {self.mm.meta[LORA_PROTOS]}"
        logger.debug(f"Binding free lora proto {with_prefix} to {self.mm.node}: {repr(lora_proto)}")
        self.mm.meta[LORA_PROTOS] = {with_prefix: lora_proto}

    @property
    def lora_protos(self) -> dict[LoraPluginInputPrefix, LoraProto]:
        """The LoRA prototypes associated with this linear layer."""
        assert (
            lora_protos := verify(self.mm.meta.get(LORA_PROTOS, {}), as_type=dict[LoraPluginInputPrefix, LoraProto])
        ) is not None
        return lora_protos

    @lora_protos.setter
    def lora_protos(self, value: dict[LoraPluginInputPrefix, LoraProto]) -> None:
        """Set the LoRA prototypes for this linear layer."""
        assert LORA_PROTOS not in self.mm.meta, f"Lora protos already set for {self.mm}: {self.mm.meta[LORA_PROTOS]}"
        self.mm.meta[LORA_PROTOS] = value

    @property
    def layer_index(self) -> int | None:
        """The layer index of this linear layer."""
        return verify(self.mm.meta.get(LAYER_INDEX), as_type=int)

    @layer_index.setter
    def layer_index(self, value: int) -> None:
        """Set the layer index for this linear layer."""
        self.mm.meta[LAYER_INDEX] = value

    @property
    def lora_prefix(self) -> LoraPluginInputPrefix | None:
        """The LoRA prefix associated with this linear layer."""
        return verify(self.mm.meta.get(LORA_PREFIX), as_type=LoraPluginInputPrefix)

    @property
    def weight_dequantize_node(self) -> Dequantize | None:
        """The weight dequantization node associated with this linear layer."""
        return Dequantize.specialize_from(self.mm.other)

    @property
    def activation_quantization(self) -> ActivationQuantization | None:
        """The activation quantization associated with this linear layer."""
        return verify(self.mm.meta.get(ACTIVATION_QUANTIZATION, None), as_type=ActivationQuantization)

    @activation_quantization.setter
    def activation_quantization(self, value: ActivationQuantization) -> None:
        """Set the activation quantization for this linear layer."""
        assert ACTIVATION_QUANTIZATION not in self.mm.meta, f"Activation quantization already set for {self.mm}"
        self.mm.meta[ACTIVATION_QUANTIZATION] = value

    def mark_expert_type_as(self, expert_type: ExpertTypeLiteral) -> None:
        """Mark the expert type of this linear layer if it is a part of a MoE layer."""
        self.mm.meta[EXPERT_TYPE] = expert_type
