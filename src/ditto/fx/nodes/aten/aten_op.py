from collections.abc import Callable
from typing import ClassVar, TypeVar

from loguru import logger
from pydantic import ConfigDict
from torch._ops import OpOverload
from torch.fx.node import Node
from typing_extensions import Self, Unpack

from ..call_function import CallFunction

ATenOpSubclass = TypeVar("ATenOpSubclass", bound="ATenOp")


class ATenOp(CallFunction):
    FINAL: ClassVar[bool] = False
    _REGISTRY: ClassVar[dict[OpOverload, type[Self]]] = {}

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        super().__init_subclass__(**kwargs)
        cls._REGISTRY = {}

    @classmethod
    def possible_targets(cls) -> tuple[OpOverload, ...]:
        assert cls._REGISTRY is not None, f"{cls.__name__} does not have final classes"
        return tuple(cls._REGISTRY.keys())

    @classmethod
    def specialize_from(cls, node: Node) -> Self | None:
        assert cls._REGISTRY is not None, f"{cls.__name__} does not have final classes"
        if not cls.FINAL and isinstance(target := node.target, OpOverload) and target in cls._REGISTRY:
            return cls._REGISTRY[target].specialize_from(node)
        return super().specialize_from(node)

    @classmethod
    def final(cls, *targets: OpOverload) -> Callable[[type[ATenOpSubclass]], type[ATenOpSubclass]]:
        assert not cls.FINAL, f"Subclasses of the final aten op class {cls.__name__} cannot be registered as an ATenOp."

        def add_to_registry(subclass: type[ATenOpSubclass]) -> type[ATenOpSubclass]:
            assert subclass.__base__ is cls, (
                f"{cls.__name__}.final method must be called on a direct subclass. "
                f"{subclass.__name__} is not a direct subclass of {cls.__name__}"
            )
            subclass._register(subclass, *targets)
            subclass.FINAL = True
            return subclass

        return add_to_registry

    @classmethod
    def _register(cls, subclass: type[ATenOpSubclass], *targets: OpOverload) -> None:
        assert issubclass(subclass, cls)
        logger.debug(f"Registering {subclass.__name__} at {cls.__name__}")
        for target in targets:
            assert not (
                target in cls._REGISTRY and cls._REGISTRY[target] is not subclass
            ), f"{target} is already registered as {cls._REGISTRY[target]} in {cls.__name__}"
            cls._REGISTRY[target] = subclass  # type: ignore[assignment]
            logger.debug(f"  * {target}")
        if (superclass := cls.__base__) and issubclass(superclass, ATenOp):
            # pylint: disable-next=no-member
            superclass._register(subclass, *targets)
