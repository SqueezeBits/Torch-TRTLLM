from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from .config import DEFAULT_DEVICE


class CacheHandler(BaseModel, ABC):
    model_config = {"arbitrary_types_allowed": True}

    @property
    @abstractmethod
    def num_hidden_layers(self) -> int:
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        ...

    @property
    @abstractmethod
    def num_key_value_heads(self) -> int:
        ...

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def is_static(self) -> bool:
        return False

    def get_shape(self, batch_size: int, seq_len: int = 0) -> tuple[int, ...]:
        return (2, self.num_hidden_layers, batch_size, self.num_key_value_heads, seq_len, self.head_dim)

    @abstractmethod
    def to_cache(self, past_key_values: torch.Tensor) -> Cache:
        ...

    def to_tensor(
        self,
        cache: Cache,
    ) -> torch.Tensor:
        if isinstance((key_cache := getattr(cache, "key_cache", None)), list) and isinstance(
            (value_cache := getattr(cache, "value_cache", None)), list
        ):
            return torch.stack((torch.stack(key_cache), torch.stack(value_cache)))
        raise NotImplementedError(
            "The default to_tensor method is only applicable to `Cache` subclass with list type attributes "
            f"`key_cache` and `value_cache`. You need to implement `to_tensor` method of {type(self)}"
            f" for converting {type(cache)} object to a single tensor by yourself."
        )

    def map_to_tensor(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args_to_replace: dict[int, Cache] = {}
        kwargs_to_replace: dict[str, Cache] = {}
        for i, arg in enumerate(args):
            if not isinstance(arg, Cache):
                continue
            args_to_replace[i] = arg
        for name, kwarg in kwargs.items():
            if not isinstance(kwarg, Cache):
                continue
            kwargs_to_replace[name] = kwarg
        args = tuple(self.to_tensor(args_to_replace[i]) if i in args_to_replace else args[i] for i in range(len(args)))
        kwargs = {
            name: self.to_tensor(kwargs_to_replace[name]) if name in kwargs_to_replace else kwargs[name]
            for name in kwargs
        }
        return (args, kwargs)

    def init_cache(
        self,
        batch_size: int,
        seq_len: int = 0,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float16,
    ) -> Cache:
        return self.to_cache(self.init_tensor(batch_size, seq_len, device=device, dtype=dtype))

    def init_tensor(
        self,
        batch_size: int,
        seq_len: int = 0,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.zeros(*self.get_shape(batch_size, seq_len), dtype=dtype, device=device)


class ConfigBasedCacheHandler(CacheHandler):
    config: PretrainedConfig

    @property
    def num_hidden_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        possible_keys = ("num_key_value_heads", "n_head", "num_heads")
        for key in possible_keys:
            if hasattr(self.config, key):
                return getattr(self.config, key)
        raise AttributeError(f"The config doesn't have attributes named {' or '.join(possible_keys)}: {self.config}")


class DynamicCacheHandler(ConfigBasedCacheHandler):
    def to_cache(self, past_key_values: torch.Tensor) -> DynamicCache:
        cache = DynamicCache()
        for i in range(past_key_values.size(1)):
            cache.update(past_key_values[0][i], past_key_values[1][i], layer_idx=i)
        return cache


class StaticCacheHandler(ConfigBasedCacheHandler):
    batch_size: int
    max_seq_len: int

    @property
    def is_static(self) -> bool:
        return True

    def to_cache(self, past_key_values: torch.Tensor) -> StaticCache:
        assert past_key_values.shape[:2] == (2, self.num_hidden_layers)
        cache = StaticCache(
            config=self.config,
            max_batch_size=self.batch_size,
            max_cache_len=self.max_seq_len,
            device=past_key_values.device,
            dtype=past_key_values.dtype,
        )
        cache.key_cache = list(past_key_values[0].unbind())
        cache.value_cache = list(past_key_values[1].unbind())
        return cache

    def init_cache(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float16,
    ) -> Cache:
        return super().init_cache(self.batch_size, self.max_seq_len, device=device, dtype=dtype)

    def init_tensor(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        *,
        device: str | torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return super().init_tensor(self.batch_size, self.max_seq_len, device=device, dtype=dtype)
