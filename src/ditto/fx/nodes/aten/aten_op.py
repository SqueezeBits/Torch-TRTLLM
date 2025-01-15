# mypy: disable-error-code=misc
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
    """Base class for ATen operator nodes.

    Attributes:
        _REGISTRY (ClassVar[dict[OpOverload, type[Self]]]): Registry mapping ATen operators to their specializations
    """

    _REGISTRY: ClassVar[dict[OpOverload, type[Self]]] = {}

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Initialize a subclass of ATenOp.

        Args:
            **kwargs: Configuration options for the subclass
        """
        super().__init_subclass__(**kwargs)
        cls._REGISTRY = {}

    @property
    def target(self) -> OpOverload:
        """Get the ATen operator target.

        Returns:
            OpOverload: The ATen operator overload
        """
        assert isinstance(op := super().target, OpOverload)
        return op

    @classmethod
    def create_from_overloadpacket(
        cls,
        graph: Graph,
        *,
        args: tuple[Argument | NodeSpecialization, ...] = (),
        kwargs: dict[str, Argument | NodeSpecialization] | None = None,
        overloadpacket: OpOverloadPacket | None = None,
    ) -> "FinalATenOp | None":
        """Create a specialized ATen operator node from an overload packet.

        When the overload packet is not specified, it will be inferred from the possible targets.

        Args:
            graph (Graph): The FX graph to create the node in
            args (tuple[Argument | NodeSpecialization, ...], optional): Positional arguments. Defaults to ().
            kwargs (dict[str, Argument | NodeSpecialization] | None, optional): Keyword arguments. Defaults to None.
            overloadpacket (OpOverloadPacket | None, optional): The ATen operator overload packet. Defaults to None.

        Returns:
            FinalATenOp | None: The created specialized node if successful, None otherwise

        Raises:
            AssertionError: If multiple overload packets are found for the ATen operator
        """
        if overloadpacket is None:
            overloadpackets = {overload.overloadpacket for overload in cls.possible_targets()}
            assert (
                len(overloadpackets) == 1
            ), f"Multiple overload packets found for {cls.__name__}. You must specify the overload packet explicitly."
            overloadpacket = overloadpackets.pop()

        overloads = [
            overload
            for overload_name in overloadpacket.overloads()
            if isinstance(overload := getattr(overloadpacket, overload_name, None), OpOverload)
        ]
        args_, kwargs_ = cls.unwrap_specialization(*args, **(kwargs or {}))
        for overload in overloads:
            if not (overload in cls._REGISTRY and issubclass(subclass := cls._REGISTRY[overload], FinalATenOp)):
                continue
            try:
                aten_op = subclass.create(graph, *args_, **kwargs_)
                logger.trace(f"Created {repr(aten_op)} from {overloadpacket}")
                return aten_op
            except (AssertionError, RuntimeError, TypeError, ValidationError):
                continue
        logger.trace(f"No specialization found for {overloadpacket} with args={args_} and kwargs={kwargs_}")
        return None

    @classmethod
    def possible_targets(cls) -> tuple[OpOverload, ...]:
        """Get all possible ATen operator targets for this specialization.

        Returns:
            tuple[OpOverload, ...]: The possible ATen operator targets
        """
        assert cls._REGISTRY, f"{cls.__name__} does not have final specializations"
        return tuple(cls._REGISTRY.keys())

    @classmethod
    def _specialize_from(cls, node: Node) -> Self:
        """Specialize from a node, looking up the appropriate specialization in the registry.

        Args:
            node (Node): The node to specialize from

        Returns:
            Self: The specialized node
        """
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
        """Register a specialization for an ATen operator target.

        Args:
            target (OpOverload): The ATen operator target to register for

        Returns:
            Callable[[type[SomeFinalATenOp]], type[SomeFinalATenOp]]: Decorator that registers the specialization
        """
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
        """Register a specialization in the registry.

        Args:
            subclass (type[SomeATenOp]): The specialization class to register
            target (OpOverload): The ATen operator target to register for
        """
        assert issubclass(subclass, cls)
        assert not (
            target in cls._REGISTRY and cls._REGISTRY[target] is not subclass
        ), f"{target} is already registered as {cls._REGISTRY[target].__name__} in {cls.__name__}"
        if cls is not FinalATenOp:
            if subclass is cls:
                logger.trace(f"{cls.__name__} is the final specialization of {str(target)}")
            else:
                logger.trace(f"{subclass.__name__} can be viewed as {cls.__name__}")
            cls._REGISTRY[target] = subclass
        if (superclass := cls.__base__) and issubclass(superclass, ATenOp):
            # pylint: disable-next=no-member
            superclass._register(subclass, target)


class FinalATenOp(ATenOp, FinalCallFunction):
    """Base class for final ATen operator specializations."""

    @classmethod
    def designated_target(cls) -> OpOverload:
        """Get the designated ATen operator target for this final specialization.

        Returns:
            OpOverload: The designated ATen operator target
        """
        assert isinstance(target := super().designated_target(), OpOverload)
        return target
