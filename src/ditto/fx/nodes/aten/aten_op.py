from collections.abc import Callable
from typing import ClassVar, TypeVar

from loguru import logger
from pydantic import ConfigDict, ValidationError
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx import Graph
from torch.fx.node import Argument, Node
from typing_extensions import Self, Unpack

from ..call_function import CallFunction, FinalCallFunction
from ..node_specialization import NodeSpecialization

SomeATenOp = TypeVar("SomeATenOp", bound="ATenOp")
SomeFinalATenOp = TypeVar("SomeFinalATenOp", bound="FinalATenOp")


class ATenOp(CallFunction):
    _REGISTRY: ClassVar[dict[OpOverload, type[Self]]] = {}

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        super().__init_subclass__(**kwargs)
        cls._REGISTRY = {}

    @property
    def target(self) -> OpOverload:
        assert isinstance(op := super().target, OpOverload)
        return op

    @classmethod
    def may_create(
        cls,
        graph: Graph,
        overload_or_packet: OpOverload | OpOverloadPacket,
        *args: Argument | NodeSpecialization,
        **kwargs: Argument | NodeSpecialization,
    ) -> "FinalATenOp | None":
        if isinstance(overload_or_packet, OpOverload):
            overloads = [overload_or_packet]
        else:
            overloads = [
                overload
                for overload_name in overload_or_packet.overloads()
                if isinstance(overload := getattr(overload_or_packet, overload_name, None), OpOverload)
            ]

        for overload in overloads:
            if not (overload in cls._REGISTRY and issubclass(subclass := cls._REGISTRY[overload], FinalATenOp)):
                continue
            try:
                return subclass.create(graph, *args, **kwargs)
            except (AssertionError, TypeError, ValidationError):
                continue
        return None

    @classmethod
    def possible_targets(cls) -> tuple[OpOverload, ...]:
        assert cls._REGISTRY, f"{cls.__name__} does not have final specializations"
        return tuple(cls._REGISTRY.keys())

    @classmethod
    def _specialize_from(cls, node: Node) -> Self:
        assert cls._REGISTRY, f"{cls.__name__} does not have final specializations"
        if (
            not issubclass(cls, FinalATenOp)
            and isinstance(target := node.target, OpOverload)
            and target in cls._REGISTRY
        ):
            return cls._REGISTRY[target]._specialize_from(node)
        return super()._specialize_from(node)

    @classmethod
    def register(cls, target: OpOverload) -> Callable[[type[SomeFinalATenOp]], type[SomeFinalATenOp]]:
        assert not issubclass(
            cls, FinalATenOp
        ), f"Subclasses of the final aten op class {cls.__name__} cannot be further registered as an ATenOp."

        def add_to_registry(subclass: type[SomeFinalATenOp]) -> type[SomeFinalATenOp]:
            assert subclass.__base__ in (
                cls,
                FinalATenOp,
            ), f"{subclass.__name__} is not a direct subclass of either {cls.__name__} or {FinalATenOp.__name__}"
            assert issubclass(subclass, ATenOp)
            subclass._register(subclass, target)
            return subclass

        return add_to_registry

    @classmethod
    def _register(cls, subclass: type[SomeATenOp], target: OpOverload) -> None:
        assert issubclass(subclass, cls)
        assert not (
            target in cls._REGISTRY and cls._REGISTRY[target] is not subclass
        ), f"{target} is already registered as {cls._REGISTRY[target].__name__} in {cls.__name__}"
        if cls is not FinalATenOp:
            if subclass is cls:
                logger.trace(f"{cls.__name__} is the final specialization of {str(target)}")
            else:
                logger.trace(f"{subclass.__name__} can be viewed as {cls.__name__}")
            cls._REGISTRY[target] = subclass  # type: ignore[assignment]
        if (superclass := cls.__base__) and issubclass(superclass, ATenOp):
            # pylint: disable-next=no-member
            superclass._register(subclass, target)


class FinalATenOp(ATenOp, FinalCallFunction):
    @classmethod
    def designated_target(cls) -> OpOverload:
        assert isinstance(target := super().designated_target(), OpOverload)
        return target
