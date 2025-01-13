from collections.abc import Callable
from typing import Any

from pydantic import Field, computed_field, model_serializer, model_validator
from typing_extensions import Self

from ...types import StrictlyTyped
from .literals import DTypeLiteral, QuantAlgoLiteral


class TRTLLMMapping(StrictlyTyped):
    """Minimal set of properties for initializing `trtllm.Mapping`."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def world_size(self) -> int:
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
        _pp_groups: list[list[int]] = []
        for i in range(self.tp_size * self.cp_size):
            ranks = range(i, self.world_size, self.tp_size * self.cp_size)
            _pp_groups.append(list(ranks))
        return _pp_groups

    @property
    def moe_tp_groups(self) -> list[list[int]]:
        _moe_tp_groups: list[list[int]] = []
        moe_tp_ep_size = self.moe_tp_size * self.moe_ep_size
        for i in range(self.pp_size):
            for j in range(self.moe_ep_size):
                ranks = range(i * moe_tp_ep_size + j, (i + 1) * moe_tp_ep_size, self.moe_ep_size)
                _moe_tp_groups.append(list(ranks))
        return _moe_tp_groups

    @property
    def moe_ep_groups(self) -> list[list[int]]:
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
        return self.rank % (self.tp_size * self.cp_size) // self.tp_size

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def pp_rank(self) -> int:
        return self.rank // (self.tp_size * self.cp_size)

    @property
    def moe_tp_rank(self) -> int:
        return self.tp_rank // self.moe_ep_size

    @property
    def moe_ep_rank(self) -> int:
        return self.tp_rank % self.moe_ep_size

    @property
    def cp_group(self) -> list[int]:
        return self.cp_groups[self.pp_rank * self.tp_size + self.tp_rank]

    @property
    def tp_group(self) -> list[int]:
        return self.tp_groups[self.pp_rank * self.cp_size + self.cp_rank]

    @property
    def pp_group(self) -> list[int]:
        return self.pp_groups[self.cp_rank * self.tp_size + self.tp_rank]

    @property
    def moe_tp_group(self) -> list[int]:
        return self.moe_tp_groups[self.pp_rank * self.moe_ep_size + self.moe_ep_rank]

    @property
    def moe_ep_group(self) -> list[int]:
        return self.moe_ep_groups[self.pp_rank * self.moe_tp_size + self.moe_tp_rank]

    @model_validator(mode="before")
    @classmethod
    def resolve_defaults_if_none(cls, data: Any) -> Any:
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
        moe_tp_ep_size = self.moe_tp_size * self.moe_ep_size
        assert moe_tp_ep_size == self.tp_size, (
            "tp_size must equal to moe_tp_size * moe_ep_size, "
            f"but got {self.tp_size} != {self.moe_tp_size} * {self.moe_ep_size}",
        )
        assert not (self.moe_ep_size != 1 and self.cp_size > 1), "CP don't support MoE tp/ep yet"

        return self

    def copy_with_rank(self, rank: int) -> Self:
        assert rank < self.world_size, "rank must be lower than world_size, " f"but got {rank} >= {self.world_size}"
        return self.__class__(**self.model_dump(), rank=rank)


class TRTLLMQuantConfig(StrictlyTyped):
    quant_algo: QuantAlgoLiteral | None = None
    kv_cache_quant_algo: QuantAlgoLiteral | None = None
    group_size: int = 128
    smoothquant_val: float = 0.5
    """Only required for SmoothQuant"""
    clamp_val: list[float] | None = Field(default=None, min_length=2, max_length=2)
    """Only required for SmoothQuant"""
    has_zero_point: bool = False
    pre_quant_scale: bool = False
    exclude_modules: list[str] | None = None


class TRTLLMPretrainedConfig(StrictlyTyped):
    """Minimal subset of properties in `trtllm.PretrainedConfig` required at runtime."""

    architecture: str
    dtype: DTypeLiteral = "float16"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    mapping: TRTLLMMapping = Field(default_factory=TRTLLMMapping)
    quantization: TRTLLMQuantConfig | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_serializer(mode="wrap")
    def serialize_model(self, original_serializer: Callable[[Self], dict[str, Any]]) -> dict[str, Any]:
        data = original_serializer(self)
        data.update(self.extra_fields)
        return data
