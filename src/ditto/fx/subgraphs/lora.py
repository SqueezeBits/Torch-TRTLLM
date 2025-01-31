import re
from typing import Literal

import tensorrt as trt
import torch
from pydantic import Field
from torch.fx import Node
from typing_extensions import Self

from ...constants import PEFT_ADAPTER_PREFIX
from ...literals import LoraStateDictSuffix
from ...types import DataType, Number, expect_identical, verify
from ..nodes import AddTensorTensor, GetAttr, MulTensorScalar, Permute, Reshape
from ..targets import LoraProto
from .linear import Linear
from .path import TrailingReformatPath
from .subgraph import Subgraph


class Lora(Subgraph):
    """A subgraph representing a single LoRA (Low-Rank Adaptation) layer.

    This subgraph identifies a pattern of two linear layers (lora_a and lora_b) that form a low-rank
    decomposition, along with a scaling factor and addition to the main model weights.

    Attributes:
        task_uid (int): Unique identifier for the LoRA task/adapter
        lora_a (Linear): First linear layer in the low-rank decomposition
        lora_b (Linear): Second linear layer in the low-rank decomposition
        mul (MulTensorScalar): Scaling operation for the LoRA output
        add (AddTensorTensor): Addition operation to combine with main weights
    """

    task_uid: int
    lora_a: Linear
    lora_b: Linear
    mul: MulTensorScalar
    add: AddTensorTensor

    def mark_as_seen(self) -> None:
        """Mark this LoRA layer as having been processed."""
        self.lora_a.mm.node.meta["seen"] = True
        self.lora_b.mm.node.meta["seen"] = True

    @property
    def is_seen(self) -> bool:
        """Whether this LoRA layer has been processed before."""
        return self.lora_a.mm.node.meta.get("seen", False) and self.lora_b.mm.node.meta.get("seen", False)

    @property
    def state_dict(self) -> dict[LoraStateDictSuffix, torch.Tensor]:
        """The state dict containing the LoRA weights."""
        return {
            "lora_A.weight": self.lora_a_weight.parameter,
            "lora_B.weight": self.lora_b_weight.parameter,
        }

    @property
    def transa(self) -> bool:
        """Whether the LoRA matrix multiplication's LHS (activation) is transposed."""
        return False

    @property
    def transb(self) -> bool:
        """Whether the LoRA matrix multiplication's RHS (weight) is transposed."""
        return Permute.specialize_from(self.lora_b.weight_node) is not None

    @property
    def input_node(self) -> Node:
        """The input node of the LoRA layer."""
        return self.lora_a.input_node

    @property
    def lora_a_weight(self) -> GetAttr:
        """The weight parameter node for the LoRA A layer."""
        assert (w := _get_lora_weight(self.lora_a.weight_node)[0]) is not None
        return w

    @property
    def lora_b_weight(self) -> GetAttr:
        """The weight parameter node for the LoRA B layer."""
        assert (w := _get_lora_weight(self.lora_b.weight_node)[0]) is not None
        return w

    @property
    def low_rank(self) -> int:
        """The low rank of the LoRA layer."""
        return self.lora_a.out_features

    @property
    def in_hidden_size(self) -> int:
        """The input hidden size of the LoRA layer."""
        return self.lora_a.in_features

    @property
    def out_hidden_size(self) -> int:
        """The output hidden size of the LoRA layer."""
        return self.lora_b.out_features

    @property
    def scaling(self) -> Number:
        """The scaling factor of the LoRA layer."""
        return self.mul.other

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (add := AddTensorTensor.specialize_from(node))
            and (mul := MulTensorScalar.specialize_from(add.other))
            and (reshape := Reshape.specialize_from(mul.this))
            and (lora_b := Linear.configure_from(reshape.this))
            and lora_b.bias_node is None
            and (lora_a := Linear.configure_from(lora_b.input_node))
            and lora_a.bias_node is None
        ):
            return None

        lora_a_weight, is_lora_a_transposed = _get_lora_weight(lora_a.weight_node)
        lora_b_weight, is_lora_b_transposed = _get_lora_weight(lora_b.weight_node)
        if not (
            (is_lora_a_transposed == is_lora_b_transposed)
            and lora_a_weight is not None
            and lora_b_weight is not None
            and lora_a.dtype == lora_b.dtype
            and lora_a.out_features == lora_b.in_features
        ):
            return None

        if (
            task_uid := expect_identical(
                get_task_uid(lora_a_weight, "lora_A"),
                get_task_uid(lora_b_weight, "lora_B"),
            )
        ) is None:
            return None

        return cls(lora_a=lora_a, lora_b=lora_b, add=add, mul=mul, task_uid=task_uid)


def get_task_uid(get_attr: GetAttr, expected_lora_type: Literal["lora_A", "lora_B"]) -> int | None:
    """Get the task uid from the original target of a GetAttr node by regular expression matching.

    Args:
        get_attr (GetAttr): A GetAttr node for a Lora weight
        expected_lora_type (Literal["lora_A", "lora_B"]): The expected type of the Lora weight

    Returns:
        int | None: The task uid if the target matches the expected pattern, None otherwise
    """
    if (
        (match := re.search(rf"(lora_[AB])\.{PEFT_ADAPTER_PREFIX}_(\d+)", get_attr.original_target))
        and (
            expect_identical(
                match.group(1),
                expected_lora_type,
                expecting_type=Literal["lora_A", "lora_B"],
            )
        )
        is not None
        and (task_uid := verify(match.group(2), as_type=int, coerce=True)) is not None
    ):
        return task_uid
    return None


class MultiLora(Subgraph):
    """A subgraph representing multiple LoRA layers applied to a single linear layer.

    This subgraph identifies a pattern of multiple LoRA layers that are applied additively
    to modify the weights of a single linear layer.

    Attributes:
        linear (Linear): The base linear layer being modified
        loras (list[Lora]): List of LoRA layers applied to the linear layer
    """

    linear: Linear
    loras: list[Lora] = Field(min_length=1)

    @property
    def input_node(self) -> Node:
        """The input node of the MultiLora layer."""
        return TrailingReformatPath.configure_from(self.linear.input_node).top

    @property
    def pre_lora_output_node(self) -> Node:
        """The output node of the MultiLora layer before the Lora layers are appended."""
        return self.loras[0].add.this

    @property
    def output_node(self) -> Node:
        """The final output node of the MultiLora layer."""
        return self.loras[-1].add.node

    @property
    def in_hidden_size(self) -> int:
        """The input hidden size of the Lora layer."""
        return self.linear.in_features

    @property
    def out_hidden_size(self) -> int:
        """The output hidden size of the Lora layer."""
        return self.linear.out_features

    @property
    def all_loras_unseen(self) -> bool:
        """Whether all Lora layers are unseen."""
        return all(not lora.is_seen for lora in self.loras)

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        loras: list[Lora] = []
        while (lora := Lora.configure_from(node)) is not None:
            loras.append(lora)
            node = lora.add.this

        if not (
            (linear := Linear.configure_from(TrailingReformatPath.configure_from(node).top))
            and len(loras) > 0
            and (
                TrailingReformatPath.configure_from(linear.input_node).top
                == TrailingReformatPath.configure_from(loras[-1].input_node).top
            )
            and all(
                lora.in_hidden_size == loras[0].in_hidden_size
                and lora.out_hidden_size == loras[0].out_hidden_size
                and lora.transa == loras[0].transa
                and lora.transb == loras[0].transb
                for lora in loras
            )
        ):
            return None
        return cls(linear=linear, loras=loras[::-1])

    @property
    def max_low_rank(self) -> int:
        """The maximum low rank among all LoRA layers."""
        return max(lora.low_rank for lora in self.loras)

    def set_free_lora_proto(self) -> None:
        """Set the free LoRA prototype for the underlying linear layer of this MultiLoRA layer.

        This configures a LoRA prototype with the combined parameters of all LoRA layers.
        """
        self.linear.free_lora_proto = LoraProto(
            max_low_rank=self.max_low_rank,
            in_hidden_size=self.in_hidden_size,
            out_hidden_size=self.out_hidden_size,
            type_id=DataType(self.linear.dtype).to(trt.DataType),
            transa=self.loras[0].transa,
            transb=self.loras[0].transb,
            state_dicts={lora.task_uid: lora.state_dict for lora in self.loras},
            # TODO: make `remove_input_padding` and `weight_index` configurable
            remove_input_padding=True,
            weight_index=0,
        )
        for lora in self.loras:
            lora.mark_as_seen()


def _get_lora_weight(weight_node: Node) -> tuple[GetAttr | None, bool]:
    """Get the LoRA weight node and whether it is transposed.

    Args:
        weight_node (Node): The weight node to examine

    Returns:
        tuple[GetAttr | None, bool]: Tuple containing:
            - The GetAttr node for the weight if found, None otherwise
            - Whether the weight is transposed
    """
    if permute := Permute.specialize_from(weight_node):
        return GetAttr.specialize_from(permute.this), True
    return GetAttr.specialize_from(weight_node), False
