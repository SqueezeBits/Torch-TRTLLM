# mypy: disable-error-code="misc"

from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorrt as trt
import torch
from pydantic import Field, field_serializer, model_serializer
from torch.fx import Graph, Node
from typing_extensions import Self

from ...arguments import DynamicDimensionType, TensorTypeHint, TRTLLMArgumentHint
from ...literals import LoraPluginInputPrefix, LoraStateDictSuffix
from ...types import DataType, StrictlyTyped
from ..nodes import Placeholder
from .fake_tensor_mode import is_in_fake_tensor_mode
from .plugin import Plugin


class LoraPluginFields(StrictlyTyped):
    """Base fields for LoRA plugin configuration.

    The order of attributes matters for TensorRT plugin field serialization.

    Attributes:
        in_hidden_size (int): Input hidden dimension size
        transa (bool): Whether to transpose first matrix in multiplication. Defaults to False.
        transb (bool): Whether to transpose second matrix in multiplication. Defaults to True.
        type_id (trt.DataType): Data type for plugin computation
        remove_input_padding (bool): Whether to remove padding from input. Defaults to True.
        max_low_rank (int): Maximum rank for low-rank adaptation
        weight_index (int): Index for weight lookup. Defaults to 0.
    """

    # the order of the attributes does matter!
    in_hidden_size: int
    transa: bool = False
    transb: bool = True
    # num_lora_modules: int should be located at here
    type_id: trt.DataType
    remove_input_padding: bool = True
    max_low_rank: int
    weight_index: int = 0


class LoraProto(LoraPluginFields):
    """Prototype class for a single LoRA plugin configuration.

    Extends LoraPluginFields with output size and state dict storage.

    Attributes:
        out_hidden_size (int): Output hidden dimension size
        state_dicts (dict[int, dict[LoraStateDictSuffix, torch.Tensor]]): State dictionaries for LoRA weights.
            Defaults to empty dict.
    """

    out_hidden_size: int
    state_dicts: dict[int, dict[LoraStateDictSuffix, torch.Tensor]] = Field(default_factory=dict, exclude=True)

    @property
    def dtype(self) -> torch.dtype:
        """Get PyTorch dtype corresponding to TensorRT type_id."""
        return DataType(self.type_id).to(torch.dtype)

    def is_compatible_with(self, other: Self) -> bool:
        """Check if this proto is compatible with another.

        Args:
            other (Self): Another LoraProto instance to compare with

        Returns:
            bool: True if configurations are compatible
        """
        return (
            self.in_hidden_size == other.in_hidden_size
            and self.dtype == other.dtype
            and self.transa == other.transa
            and self.transb == other.transb
            and self.remove_input_padding == other.remove_input_padding
            and self.weight_index == other.weight_index
        )

    @field_serializer("type_id")  # Required for writing at ONNX node metadata
    def serialize_type_id(self, type_id: trt.DataType) -> str:
        """Serialize TensorRT DataType to string.

        Args:
            type_id (trt.DataType): TensorRT data type

        Returns:
            str: Lowercase string representation of data type
        """
        return type_id.name.lower()


class LoraPlugin(Plugin, LoraPluginFields):
    """TensorRT plugin implementation for multi-LoRA (Low-Rank Adaptation).

    Attributes:
        out_hidden_sizes (list[int]): List of output hidden dimension sizes
    """

    out_hidden_sizes: list[int] = Field(exclude=True)

    @property
    def num_lora_modules(self) -> int:
        """Get number of LoRA modules.

        Returns:
            int: Number of LoRA modules based on output sizes
        """
        return len(self.out_hidden_sizes)

    @property
    def dtype(self) -> torch.dtype:
        """PyTorch dtype corresponding to TensorRT type_id."""
        return DataType(self.type_id).to(torch.dtype)

    @classmethod
    def get_field_dtype(cls, name: str, value: Any) -> type[np.number]:
        """Get numpy dtype for serializing plugin fields.

        Args:
            name (str): Field name
            value (Any): Field value

        Returns:
            type[np.number]: Numpy dtype for field serialization
        """
        if name == "remove_input_padding":
            return np.int8
        if name in {"transa", "transb"}:
            return np.int32
        return super().get_field_dtype(name, value)

    def get_fields(self) -> list[trt.PluginField]:
        """Get TensorRT plugin fields for serialization.

        Returns:
            list[trt.PluginField]: List of plugin fields
        """
        fields = super().get_fields()
        fields.insert(3, self.as_plugin_field("num_lora_modules", self.num_lora_modules))
        fields.extend(
            self.as_plugin_field(f"out_hidden_size_{i}", value) for i, value in enumerate(self.out_hidden_sizes)
        )
        return fields

    @property
    def out_hidden_size(self) -> int:
        """Total output hidden size."""
        return sum(self.out_hidden_sizes)

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply LoRA to input tensor.

        Args:
            x (torch.Tensor): Input tensor
            **kwargs (Any): Additional keyword arguments

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Output tensor(s) after applying LoRA

        Raises:
            NotImplementedError: If not in fake tensor mode
        """
        if is_in_fake_tensor_mode():
            # Note that this is merely for the fake tensor propagation
            if self.num_lora_modules == 1:
                return torch._C._nn.linear(x, torch.empty(self.out_hidden_sizes[0], self.in_hidden_size))
            return tuple(
                torch._C._nn.linear(x, torch.empty(out_hidden_size, self.in_hidden_size))
                for out_hidden_size in self.out_hidden_sizes
            )
        raise NotImplementedError(f"{type(self).__name__} doesn't have implementation")


class LoraPluginInputPair(StrictlyTyped):
    """Pair of input placeholders for LoRA plugin.

    Attributes:
        ranks (Placeholder): Placeholder for rank values
        weights_pointers (Placeholder): Placeholder for weight pointers
        rank_hint (TensorTypeHint): Type hint for rank tensor
        weights_pointer_hint (TensorTypeHint): Type hint for weight pointer tensor
    """

    ranks: Placeholder
    weights_pointers: Placeholder
    rank_hint: TensorTypeHint
    weights_pointer_hint: TensorTypeHint

    @property
    def hints(self) -> dict[str, TensorTypeHint]:
        """Type hints for input tensors."""
        return {self.ranks.target: self.rank_hint, self.weights_pointers.target: self.weights_pointer_hint}

    @classmethod
    def create(
        cls,
        graph: Graph,
        batch_size: int | DynamicDimensionType,
        prefix: LoraPluginInputPrefix,
        layer_idx: int,
    ) -> Self:
        """Create input placeholders in graph.

        Args:
            graph (Graph): FX graph to add nodes to
            batch_size (int | DynamicDimensionType): Batch size or dynamic dimension
            prefix (LoraPluginInputPrefix): Prefix for input names
            layer_idx (int): Layer index

        Returns:
            Self: New LoraPluginInputPair instance

        Raises:
            RuntimeError: If no placeholder found in graph
        """
        if not (placeholders := list(graph.find_nodes(op="placeholder"))):
            raise RuntimeError("No placeholder found in the graph")
        last_placeholder = placeholders[-1]
        with graph.inserting_after(last_placeholder):
            weights_pointers = Placeholder.create(
                graph,
                f"{prefix}_lora_weights_pointers_{layer_idx}",
                hint=(weights_pointer_hint := TensorTypeHint(dtype=torch.int64, shape=(batch_size, 2))),
            )
            ranks = Placeholder.create(
                graph,
                f"{prefix}_lora_ranks_{layer_idx}",
                hint=(rank_hint := TensorTypeHint(dtype=torch.int32, shape=(batch_size,))),
            )
        return cls(
            ranks=ranks,
            weights_pointers=weights_pointers,
            rank_hint=rank_hint,
            weights_pointer_hint=weights_pointer_hint,
        )


class LoraPluginInputs(StrictlyTyped):
    """Collection of inputs for LoRA plugin.

    Attributes:
        host_request_types (Node): Node for request type inputs
        input_pairs (list[LoraPluginInputPair]): List of input pairs. Defaults to empty list.
        host_context_lengths (Node): Node for context length inputs
    """

    host_request_types: Node
    input_pairs: list[LoraPluginInputPair] = Field(default_factory=list)
    host_context_lengths: Node

    @property
    def num_lora_modules(self) -> int:
        """Number of LoRA modules."""
        return len(self.input_pairs)

    @classmethod
    def collect_from(cls, graph: Graph, input_pairs: list[LoraPluginInputPair]) -> Self | None:
        """Collect existing inputs from graph.

        Args:
            graph (Graph): FX graph to search
            input_pairs (list[LoraPluginInputPair]): List of input pairs

        Returns:
            Self | None: New instance if required nodes found, None otherwise
        """
        if not (
            (hrt_candidates := list(graph.find_nodes(op="placeholder", target="host_request_types")))
            and (hcl_candidates := list(graph.find_nodes(op="placeholder", target="host_context_lengths")))
            and len(hrt_candidates) == 1
            and len(hcl_candidates) == 1
        ):
            return None
        host_request_types = hrt_candidates[0]
        host_context_lengths = hcl_candidates[0]
        return cls(
            host_request_types=host_request_types,
            input_pairs=input_pairs,
            host_context_lengths=host_context_lengths,
        )

    @classmethod
    def create_and_sync(
        cls,
        graph: Graph,
        argument_hint: TRTLLMArgumentHint,
        prefixes: Sequence[LoraPluginInputPrefix],
        layer_index: int,
    ) -> Self | None:
        """Create new inputs and sync with argument hints.

        Args:
            graph (Graph): FX graph to modify
            argument_hint (TRTLLMArgumentHint): Argument hints to update
            prefixes (Sequence[LoraPluginInputPrefix]): Input prefixes to create
            layer_index (int): Layer index

        Returns:
            Self | None: New instance if required nodes found, None otherwise
        """
        if not (
            (hrt_candidates := list(graph.find_nodes(op="placeholder", target="host_request_types")))
            and (hcl_candidates := list(graph.find_nodes(op="placeholder", target="host_context_lengths")))
            and len(hrt_candidates) == 1
            and len(hcl_candidates) == 1
        ):
            return None
        host_request_types = hrt_candidates[0]
        host_context_lengths = hcl_candidates[0]
        inputs = cls(
            host_request_types=host_request_types,
            host_context_lengths=host_context_lengths,
        )
        for prefix in prefixes:
            inputs.append_input_pairs_and_sync(graph, argument_hint, prefix, layer_index)
        return inputs

    def append_input_pairs_and_sync(
        self,
        graph: Graph,
        argument_hint: TRTLLMArgumentHint,
        prefix: LoraPluginInputPrefix,
        layer_index: int,
    ) -> None:
        """Append Lora plugin input nodes to the graph and the argument hints.

        This method creates placeholder nodes for a single Lora prefix and updates the necessary
        hints and internal state. Specifically, it:
        1. Creates placeholder nodes in the graph for the Lora inputs
        2. Updates the argument hints with the new Lora input hints
        3. Appends the new Lora input pair to the `input_pairs` list

        The newly added Lora input pairs will be used by TRT-LLM Lora plugin nodes.

        Args:
            graph: The FX graph being modified. New placeholder nodes will be added.
            argument_hint: Type hints for TRTLLM inputs, containing batch size and other specs.
                Will be updated with the new Lora input hints.
            prefix: Position in the transformer layer predefined by TRT-LLM to add Lora to
                (e.g. "attn_q", "attn_k", "attn_v" or "attn_dense"). See LoraPluginInputPrefix for all supported values.
            layer_index: Index of the current transformer layer, used for unique input names.
        """
        input_pair = LoraPluginInputPair.create(
            graph=graph,
            batch_size=argument_hint.batch_size,
            prefix=prefix,
            layer_idx=layer_index,
        )
        argument_hint.lora_input_hints.update(input_pair.hints)
        self.input_pairs.append(input_pair)

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        """Serialize model to dictionary format.

        Returns:
            dict[str, Any]: Serialized model with host request types, rank inputs, weight pointer inputs,
                and host context lengths
        """
        return {
            "host_request_types": self.host_request_types,
            # All the rank inputs must appear before the weights pointer inputs
            **{input_pair.ranks.target: input_pair.ranks.node for input_pair in self.input_pairs},
            **{input_pair.weights_pointers.target: input_pair.weights_pointers.node for input_pair in self.input_pairs},
            "host_context_lengths": self.host_context_lengths,
        }
