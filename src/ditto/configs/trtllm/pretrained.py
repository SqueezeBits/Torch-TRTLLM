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

# mypy: disable-error-code="misc"

from collections.abc import Callable
from typing import Any

from pydantic import Field, computed_field, model_serializer, model_validator
from typing_extensions import Self

from ...literals import DTypeLiteral, QuantAlgoLiteral
from ...types import StrictlyTyped


class TRTLLMMapping(StrictlyTyped):
    """Minimal set of properties for initializing `trtllm.Mapping`.

    Handles tensor parallel (TP), pipeline parallel (PP), and MoE parallel configurations.

    Attributes:
        gpus_per_node (int): Number of GPUs per node. Defaults to 8.
        cp_size (int): Size of checkpoint parallel dimension. Defaults to 1.
        tp_size (int): Size of tensor parallel dimension. Defaults to 1.
        pp_size (int): Size of pipeline parallel dimension. Defaults to 1.
        moe_tp_size (int): Size of MoE tensor parallel dimension. Defaults to 0.
        moe_ep_size (int): Size of MoE expert parallel dimension. Defaults to 0.
        rank (int): Current process rank. Defaults to 0.
    """

    @computed_field
    @property
    def world_size(self) -> int:
        """Calculate total world size from parallel dimensions.

        Returns:
            int: Total number of processes (cp_size * tp_size * pp_size)
        """
        return self.cp_size * self.tp_size * self.pp_size

    gpus_per_node: int = Field(default=8, ge=1)
    cp_size: int = Field(default=1, ge=1)
    tp_size: int = Field(default=1, ge=1)
    pp_size: int = Field(default=1, ge=1)
    moe_tp_size: int = Field(default=0)
    moe_ep_size: int = Field(default=0)
    rank: int = Field(default=0, exclude=True)

    @property
    def cp_groups(self) -> list[list[int]]:
        """Get checkpoint parallel process groups.

        Returns:
            list[list[int]]: List of process groups for checkpoint parallelism
        """
        _cp_groups: list[list[int]] = []
        for i in range(self.pp_size):
            for j in range(self.tp_size):
                ranks = range(
                    i * self.tp_size * self.cp_size + j, (i + 1) * self.tp_size * self.cp_size + j, self.tp_size
                )
                _cp_groups.append(list(ranks))
        return _cp_groups

    @property
    def tp_groups(self) -> list[list[int]]:
        """Get tensor parallel process groups.

        Returns:
            list[list[int]]: List of process groups for tensor parallelism
        """
        _tp_groups: list[list[int]] = []
        for i in range(self.pp_size):
            for j in range(self.cp_size):
                ranks = range(
                    i * self.tp_size * self.cp_size + j * self.tp_size,
                    i * self.tp_size * self.cp_size + (j + 1) * self.tp_size,
                )
                _tp_groups.append(list(ranks))
        return _tp_groups

    @property
    def pp_groups(self) -> list[list[int]]:
        """Get pipeline parallel process groups.

        Returns:
            list[list[int]]: List of process groups for pipeline parallelism
        """
        _pp_groups: list[list[int]] = []
        for i in range(self.tp_size * self.cp_size):
            ranks = range(i, self.world_size, self.tp_size * self.cp_size)
            _pp_groups.append(list(ranks))
        return _pp_groups

    @property
    def moe_tp_groups(self) -> list[list[int]]:
        """Get MoE tensor parallel process groups.

        Returns:
            list[list[int]]: List of process groups for MoE tensor parallelism
        """
        _moe_tp_groups: list[list[int]] = []
        moe_tp_ep_size = self.moe_tp_size * self.moe_ep_size
        for i in range(self.pp_size):
            for j in range(self.moe_ep_size):
                ranks = range(i * moe_tp_ep_size + j, (i + 1) * moe_tp_ep_size, self.moe_ep_size)
                _moe_tp_groups.append(list(ranks))
        return _moe_tp_groups

    @property
    def moe_ep_groups(self) -> list[list[int]]:
        """Get MoE expert parallel process groups.

        Returns:
            list[list[int]]: List of process groups for MoE expert parallelism
        """
        _moe_ep_groups: list[list[int]] = []
        moe_tp_ep_size = self.moe_tp_size * self.moe_ep_size
        for i in range(self.pp_size):
            for j in range(self.moe_tp_size):
                ranks = range(
                    i * moe_tp_ep_size + j * self.moe_ep_size, i * moe_tp_ep_size + (j + 1) * self.moe_ep_size
                )
                _moe_ep_groups.append(list(ranks))
        return _moe_ep_groups

    @property
    def cp_rank(self) -> int:
        """Get checkpoint parallel rank.

        Returns:
            int: Current process's checkpoint parallel rank
        """
        return self.rank % (self.tp_size * self.cp_size) // self.tp_size

    @property
    def tp_rank(self) -> int:
        """Get tensor parallel rank.

        Returns:
            int: Current process's tensor parallel rank
        """
        return self.rank % self.tp_size

    @property
    def pp_rank(self) -> int:
        """Get pipeline parallel rank.

        Returns:
            int: Current process's pipeline parallel rank
        """
        return self.rank // (self.tp_size * self.cp_size)

    @property
    def moe_tp_rank(self) -> int:
        """Get MoE tensor parallel rank.

        Returns:
            int: Current process's MoE tensor parallel rank
        """
        return self.tp_rank // self.moe_ep_size

    @property
    def moe_ep_rank(self) -> int:
        """Get MoE expert parallel rank.

        Returns:
            int: Current process's MoE expert parallel rank
        """
        return self.tp_rank % self.moe_ep_size

    @property
    def cp_group(self) -> list[int]:
        """Get current process's checkpoint parallel group.

        Returns:
            list[int]: Process ranks in the current CP group
        """
        return self.cp_groups[self.pp_rank * self.tp_size + self.tp_rank]

    @property
    def tp_group(self) -> list[int]:
        """Get current process's tensor parallel group.

        Returns:
            list[int]: Process ranks in the current TP group
        """
        return self.tp_groups[self.pp_rank * self.cp_size + self.cp_rank]

    @property
    def pp_group(self) -> list[int]:
        """Get current process's pipeline parallel group.

        Returns:
            list[int]: Process ranks in the current PP group
        """
        return self.pp_groups[self.cp_rank * self.tp_size + self.tp_rank]

    @property
    def moe_tp_group(self) -> list[int]:
        """Get current process's MoE tensor parallel group.

        Returns:
            list[int]: Process ranks in the current MoE TP group
        """
        return self.moe_tp_groups[self.pp_rank * self.moe_ep_size + self.moe_ep_rank]

    @property
    def moe_ep_group(self) -> list[int]:
        """Get current process's MoE expert parallel group.

        Returns:
            list[int]: Process ranks in the current MoE EP group
        """
        return self.moe_ep_groups[self.pp_rank * self.moe_tp_size + self.moe_tp_rank]

    @model_validator(mode="before")
    @classmethod
    def resolve_defaults_if_none(cls, data: Any) -> Any:
        """Resolve default values for MoE parallel dimensions if not specified.

        Args:
            data (Any): Input configuration data

        Returns:
            Any: Configuration with resolved MoE parallel dimensions
        """
        if isinstance(data, dict):
            tp_size = data.get("tp_size", 1)
            moe_tp_size = data.get("moe_tp_size", None)
            moe_ep_size = data.get("moe_ep_size", None)
            if moe_tp_size is None and moe_ep_size is None:
                moe_tp_size = tp_size
                moe_ep_size = 1
            elif moe_tp_size is None:
                moe_tp_size = tp_size // moe_ep_size
            elif moe_ep_size is None:
                moe_ep_size = tp_size // moe_tp_size

            data["moe_tp_size"] = moe_tp_size
            data["moe_ep_size"] = moe_ep_size

        return data

    @model_validator(mode="after")
    def verify_init(self) -> Self:
        """Verify parallel configuration is valid.

        Returns:
            Self: Validated configuration instance

        Raises:
            AssertionError: If parallel dimensions are invalid
        """
        moe_tp_ep_size = self.moe_tp_size * self.moe_ep_size
        assert moe_tp_ep_size == self.tp_size, (
            "tp_size must equal to moe_tp_size * moe_ep_size, "
            f"but got {self.tp_size} != {self.moe_tp_size} * {self.moe_ep_size}",
        )
        assert not (self.moe_ep_size != 1 and self.cp_size > 1), "CP don't support MoE tp/ep yet"

        return self

    def copy_with_rank(self, rank: int) -> Self:
        """Create a copy of configuration with new rank.

        Args:
            rank (int): New process rank

        Returns:
            Self: New configuration instance with specified rank

        Raises:
            AssertionError: If rank is invalid
        """
        assert rank < self.world_size, f"rank must be lower than world_size, but got {rank} >= {self.world_size}"
        return self.__class__(**self.model_dump(), rank=rank)


class TRTLLMQuantConfig(StrictlyTyped):
    """Configuration for model quantization in TRT-LLM.

    Attributes:
        quant_algo (QuantAlgoLiteral | None): Quantization algorithm. Defaults to None.
        kv_cache_quant_algo (QuantAlgoLiteral | None): KV cache quantization algorithm. Defaults to None.
        group_size (int): Size of quantization groups. Defaults to 128.
        smoothquant_val (float): SmoothQuant alpha parameter. Defaults to 0.5.
        clamp_val (list[float] | None): Min/max clamp values for SmoothQuant. Defaults to None.
        has_zero_point (bool): Whether quantization includes zero point. Defaults to False.
        pre_quant_scale (bool): Whether to apply scaling before quantization. Defaults to False.
        exclude_modules (list[str] | None): Module names to exclude from quantization. Defaults to None.
    """

    quant_algo: QuantAlgoLiteral | None = None
    kv_cache_quant_algo: QuantAlgoLiteral | None = None
    group_size: int = 128
    smoothquant_val: float = 0.5
    clamp_val: list[float] | None = Field(default=None, min_length=2, max_length=2)
    has_zero_point: bool = False
    pre_quant_scale: bool = False
    exclude_modules: list[str] | None = None


class TRTLLMPretrainedConfig(StrictlyTyped):
    """Configuration for pretrained models in TRT-LLM.

    Attributes:
        architecture (str): Model architecture name
        dtype (DTypeLiteral): Model data type. Defaults to "float16".
        vocab_size (int): Size of vocabulary
        hidden_size (int): Size of hidden layers
        num_hidden_layers (int): Number of hidden layers
        num_attention_heads (int): Number of attention heads
        num_key_value_heads (int): Number of key/value heads
        intermediate_size (int): Size of intermediate layers
        mapping (TRTLLMMapping): Parallel mapping configuration. Defaults to TRTLLMMapping().
        quantization (TRTLLMQuantConfig | None): Quantization configuration. Defaults to None.
        extra_fields (dict[str, Any]): Additional configuration fields. Defaults to empty dict.
    """

    architecture: str
    dtype: DTypeLiteral = "float16"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    mapping: TRTLLMMapping = Field(default_factory=TRTLLMMapping)
    quantization: TRTLLMQuantConfig | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_serializer(mode="wrap")
    def serialize_model(self, original_serializer: Callable[[Self], dict[str, Any]]) -> dict[str, Any]:
        """Serialize model configuration including extra fields.

        Args:
            original_serializer (Callable[[Self], dict[str, Any]]): Original serialization function

        Returns:
            dict[str, Any]: Serialized configuration with extra fields
        """
        data = original_serializer(self)
        data.update(self.extra_fields)
        return data
