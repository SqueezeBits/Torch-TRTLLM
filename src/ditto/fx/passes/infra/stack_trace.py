# pylint: disable=unused-argument, invalid-name

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Generic

import torch
from torch.fx import Node
from torch.fx.graph import _parse_stack_trace, _ParsedStackTrace
from typing_extensions import Self, TypeVar

from ....literals import LoraPluginInputPrefix
from ....types import StrictlyTyped, verify
from ...metadata_keys import LORA_PROTOS, ORIGINAL_TARGET, STACK_TRACE, TENSOR_META, VAL
from ...nodes import NodeSpecialization
from ...targets import LoraProto


class StackTrace(_ParsedStackTrace):
    """A parsed stack trace with additional functionality.

    Extends _ParsedStackTrace to add raw string representation and parsing capabilities.
    """

    @property
    def raw(self) -> str:
        """Get the raw stack trace string."""
        return f'File "{self.file}", line {self.lineno}, in {self.name}\n    Created by {self.code}'

    @classmethod
    def parse(cls, stack_trace: str | None) -> Self | None:
        """Parse a stack trace string into a StackTrace object.

        Args:
            stack_trace: The stack trace string to parse

        Returns:
            A StackTrace object if parsing succeeds, None otherwise
        """
        if stack_trace is None or (_self := _parse_stack_trace(stack_trace)) is None:
            return None
        return cls(
            file=_self.file,
            lineno=_self.lineno,
            name=_self.name,
            code=_self.code,
        )


def inject_stack_trace_from(
    node: Node | NodeSpecialization,
    *others: Node | NodeSpecialization,
    to: Node | NodeSpecialization,
) -> None:
    """Inject stack trace and other metadata from source node(s) to target node.

    Args:
        node: The primary source node
        others: Additional source nodes
        to: The target node to inject metadata into
    """
    for propagator_class in PROPAGATOR_REGISTRY.values():
        propagator_class.update_metadata(node, *others, to=to)
    # Simply copy all non-registered metadata to the target node.
    to.meta.update({name: value for name, value in node.meta.items() if name not in PROPAGATOR_REGISTRY})


PropagatorType = TypeVar("PropagatorType", bound="MetadataPropagator")
MetadataType = TypeVar("MetadataType", default=None)
PROPAGATOR_REGISTRY: dict[str, type["MetadataPropagator"]] = {}


class MetadataPropagator(StrictlyTyped, ABC, Generic[MetadataType]):
    """Base class for metadata propagators that handle transferring metadata between nodes.

    Attributes:
        KEY: Class variable storing the metadata key this propagator handles
    """

    KEY: ClassVar[str]

    @classmethod
    def register(cls, key: str) -> Callable[[type[PropagatorType]], type[PropagatorType]]:
        """Register a metadata propagator for a specific key.

        Args:
            key: The metadata key to register for

        Returns:
            A decorator that registers the propagator class
        """

        def decorator(subclass: type[PropagatorType]) -> type[PropagatorType]:
            PROPAGATOR_REGISTRY[key] = subclass
            subclass.KEY = key
            return subclass

        return decorator

    @classmethod
    def update_metadata(
        cls,
        node: Node | NodeSpecialization,
        *others: Node | NodeSpecialization,
        to: Node | NodeSpecialization,
    ) -> None:
        """Update metadata on the target node from source node(s).

        Args:
            node (Node | NodeSpecialization): The primary source node
            others (Node | NodeSpecialization): Additional source nodes
            to (Node | NodeSpecialization): The target node to update metadata on
        """
        if (value := cls.compute_metadata(node, *others, to=to)) is not None:
            to.meta[cls.KEY] = value

    @classmethod
    @abstractmethod
    def compute_metadata(
        cls,
        node: Node | NodeSpecialization,
        *others: Node | NodeSpecialization,
        to: Node | NodeSpecialization,
    ) -> MetadataType | None:
        """Compute metadata for the target node from source node(s).

        Args:
            node (Node | NodeSpecialization): The primary source node
            others (Node | NodeSpecialization): Additional source nodes
            to (Node | NodeSpecialization): The target node to compute metadata for

        Returns:
            The computed metadata or None if no metadata is computed
        """


@MetadataPropagator.register(TENSOR_META)
@MetadataPropagator.register(VAL)
class Forget(MetadataPropagator[None]):
    """Propagator that forgets tensor metadata and values by returning None."""

    @classmethod
    def compute_metadata(
        cls,
        node: Node | NodeSpecialization,
        *others: Node | NodeSpecialization,
        to: Node | NodeSpecialization,
    ) -> None:
        """Forget tensor metadata and values by returning None.

        Args:
            node (Node | NodeSpecialization): The source node
            others (Node | NodeSpecialization): Additional source nodes
            to (Node | NodeSpecialization): The target node
        """
        return None


@MetadataPropagator.register(STACK_TRACE)
class AppendStackTrace(MetadataPropagator[str]):
    """Propagator that appends stack trace information from source to target nodes."""

    @classmethod
    def compute_metadata(
        cls,
        node: Node | NodeSpecialization,
        *others: Node | NodeSpecialization,
        to: Node | NodeSpecialization,
    ) -> str | None:
        """Append stack trace information from source to target nodes.

        Args:
            node (Node | NodeSpecialization): The source node
            others (Node | NodeSpecialization): Additional source nodes
            to (Node | NodeSpecialization): The target node

        Returns:
            str | None: The appended stack trace or None if no stack trace is available
        """
        if node.stack_trace is None:
            return None
        if parsed_stack_trace := StackTrace.parse(to.stack_trace):
            code = parsed_stack_trace.code
            if others:
                code = f"{code} fusing ({', '.join(n.name for n in (node, *others))})"
            else:
                code = f"{code} substituting {node.name}"
            return f"{node.stack_trace} -> {code}"
        return node.stack_trace


@MetadataPropagator.register(ORIGINAL_TARGET)
class KeepOriginalTargetForGetAttr(MetadataPropagator[str]):
    """Propagator that preserves the original target for get_attr operations."""

    @classmethod
    def compute_metadata(
        cls,
        node: Node | NodeSpecialization,
        *others: Node | NodeSpecialization,
        to: Node | NodeSpecialization,
    ) -> str | None:
        """Preserve the original target for get_attr operations.

        Args:
            node (Node | NodeSpecialization): The source node
            others (Node | NodeSpecialization): Additional source nodes
            to (Node | NodeSpecialization): The target node

        Returns:
            str | None: The original target or None if the node is not a get_attr operation
        """
        if node.op == to.op == "get_attr" and not others:
            return node.meta.get(cls.KEY, node.target)
        return None


@MetadataPropagator.register(LORA_PROTOS)
class MergeLoraProtos(MetadataPropagator[dict[LoraPluginInputPrefix, LoraProto]]):
    """Propagator that merges LoRA prototypes from multiple mm nodes."""

    @classmethod
    def compute_metadata(
        cls,
        node: Node | NodeSpecialization,
        *others: Node | NodeSpecialization,
        to: Node | NodeSpecialization,
    ) -> dict[LoraPluginInputPrefix, LoraProto] | None:
        """Merge LoRA prototypes from multiple mm nodes.

        Args:
            node (Node | NodeSpecialization): The source node
            others (Node | NodeSpecialization): Additional source nodes
            to (Node | NodeSpecialization): The target node

        Returns:
            dict[LoraPluginInputPrefix, LoraProto] | None: The merged LoRA prototypes or None
                if the nodes are not mm nodes
        """
        if all(n.target is not torch.ops.aten.mm.default for n in (node, *others, to)):
            return None
        lora_protos: dict[LoraPluginInputPrefix, LoraProto] = {}
        for n in (node, *others):
            if (protos := verify(n.meta.get(LORA_PROTOS), as_type=dict[LoraPluginInputPrefix, LoraProto])) is not None:
                lora_protos.update(protos)
        return lora_protos
