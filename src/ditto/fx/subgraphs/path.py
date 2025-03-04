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

# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
import sys
from typing import TypeVar, overload

import torch
from torch._ops import OpOverload
from torch.fx import Node
from typing_extensions import Self

from ...types import NodeCriterion
from ..nodes import NodeSpecialization
from ..utils import get_tensor_metadata
from .subgraph import Subgraph


class Path(Subgraph):
    """A generic representation of a path in a computational graph.

    This class models a sequence of nodes in a computational graph, sorted from the
    bottom-most node to the top-most node. It provides methods to extract and traverse
    the path.

    Attributes:
        node_seq (tuple[Node, ...]): The sequence of nodes in the path, sorted from
            bottom to top.
    """

    node_seq: tuple[Node, ...]  # sorted from bottom to top

    @property
    def nodes(self) -> list[Node]:
        """The sequence of nodes as a list."""
        return list(self.node_seq)

    @property
    def top(self) -> Node:
        """The top-most node in the path."""
        return self.node_seq[-1]

    @property
    def bottom(self) -> Node:
        """The bottom-most node in the path."""
        return self.node_seq[0]

    @classmethod
    def configure_from(
        cls, node: Node, *, break_if: NodeCriterion = lambda _: False, max_len: int = sys.maxsize
    ) -> Self:
        """Configure a path starting from a given node.

        Traverses the computational graph upward from the specified node to construct
        a path. The traversal stops when a condition is met, such as a custom break
        criterion or reaching the maximum path length.

        Args:
            node (Node): The starting node for the path.
            break_if (NodeCriterion, optional): A function to determine when to stop
                the traversal. Defaults to a lambda that always returns False.
            max_len (int, optional): The maximum length of the path. Defaults to sys.maxsize.

        Returns:
            Path: The configured path object.
        """
        nodes: list[Node] = []
        top = node

        while len(nodes) < max_len:
            nodes.append(top)
            if len(parents := get_parents(top)) != 1 or break_if(top):
                break
            top = parents[0]

        return cls(node_seq=tuple(nodes))


def get_parents(node: Node) -> list[Node]:
    """Retrieve the parent nodes of a given node.

    Filters out nodes corresponding to specific operations like size computation.

    Args:
        node (Node): The node for which to find parent nodes.

    Returns:
        list[Node]: A list of parent nodes.
    """
    return [n for n in node.all_input_nodes if n.target is not torch.ops.aten.sym_size.int]


# pylint: disable-next=invalid-name
NodeType = TypeVar("NodeType", bound=NodeSpecialization)
# pylint: disable-next=invalid-name
SubgraphType = TypeVar("SubgraphType", bound=Subgraph)


class TrailingReformatPath(Path):
    """A path specialized for trailing reformat operations in computational graphs.

    This class represents a sequence of reformat operations (e.g., reshape, permute)
    trailing from a specified node in the computational graph. It provides utilities
    to analyze and manage reformat operations.
    """

    @property
    def reformats(self) -> tuple[Node, ...]:
        """The sequence of reformat nodes, excluding the top-most node."""
        return self.node_seq[:-1]

    @classmethod
    def get_reformat_targets(cls) -> tuple[OpOverload, ...]:
        """Get the operations classified as reformat targets.

        Returns:
            tuple[OpOverload, ...]: A tuple of operations such as reshape, permute, and squeeze.
        """
        return (
            torch.ops.aten._to_copy.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.permute.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.aten.unsqueeze.default,
        )

    @classmethod
    def configure_from(
        cls, node: Node, *, break_if: NodeCriterion = lambda _: False, max_len: int = sys.maxsize
    ) -> Self:
        """Configure a trailing reformat path from a given node.

        Traverses upward from the specified node to identify a sequence of reformat
        operations. The traversal stops based on a break criterion or maximum length.

        Args:
            node (Node): The starting node for the reformat path.
            break_if (NodeCriterion, optional): A function to determine when to stop traversal.
                Defaults to a lambda that always returns False.
            max_len (int, optional): The maximum length of the path. Defaults to sys.maxsize.

        Returns:
            TrailingReformatPath: The configured reformat path object.
        """
        return super().configure_from(
            node, break_if=lambda n: n.target not in cls.get_reformat_targets() or break_if(n), max_len=max_len
        )

    @property
    def total_expansion(self) -> int | None:
        """The total expansion factor of the trailing reformat path.

        Calculated as the ratio of the number of elements in the bottom-most and top-most nodes.
        """
        if not self.reformats:
            return 1
        if not ((top := get_tensor_metadata(self.top)) and (bottom := get_tensor_metadata(self.bottom))):
            return None
        # Note that `torch.ops.aten.expand.default` is the only target that can increase the number of elements.
        # A naive implementation would be simple:
        # `return bottom.shape.numel() // top.shape.numel()`
        # However, this naive implementation involves direct division on `torch.SymInt` objects,
        # adding unwanted shape constraints to the ambient shape environment.
        # As a result, it can affect the `FakeTensorProp` to believe some of the `torch.SymInt` objects are constant.
        # Therefore, we must handle `int` and `torch.SymInt` objects separately here.
        bottom_sym_shape = [s for s in bottom.shape if isinstance(s, torch.SymInt)]
        top_sym_shape = [s for s in top.shape if isinstance(s, torch.SymInt)]
        if are_same_as_sets(bottom_sym_shape, top_sym_shape):
            return None

        bottom_ishape = torch.Size(s for s in bottom.shape if isinstance(s, int))
        top_ishape = torch.Size(s for s in top.shape if isinstance(s, int))
        return bottom_ishape.numel() // top_ishape.numel()

    @classmethod
    @overload
    def traceback(
        cls,
        node_type: type[NodeType],
        node: Node,
        *,
        break_if: NodeCriterion = lambda _: False,
    ) -> NodeType | None:
        ...

    @classmethod
    @overload
    def traceback(
        cls,
        node_type: type[SubgraphType],
        node: Node,
        *,
        break_if: NodeCriterion = lambda _: False,
    ) -> SubgraphType | None:
        ...

    @classmethod
    def traceback(
        cls,
        node_type: type[NodeType] | type[SubgraphType],
        node: Node,
        *,
        break_if: NodeCriterion = lambda _: False,
    ) -> NodeType | SubgraphType | None:
        """Traceback to find a specific node type in the trailing reformat path.

        Args:
            node_type (type[NodeType] | type[Subgraph]): The type of node or subgraph to search for.
            node (Node): The starting node for the traceback.
            break_if (NodeCriterion, optional): A function to determine when to stop tracing.
                Defaults to a lambda that always returns False.

        Returns:
            NodeType | Subgraph | None: The specialized node/subgraph of the given type, or None if not found.
        """
        if issubclass(node_type, Subgraph):
            return node_type.configure_from(TrailingReformatPath.configure_from(node, break_if=break_if).top)
        return node_type.specialize_from(TrailingReformatPath.configure_from(node, break_if=break_if).top)

    @classmethod
    def get_parents(cls, node: Node) -> list[Node]:
        """Get parent nodes, skipping over reformat operations.

        This method recursively traverses up the graph to find parent nodes, skipping over any
        reformat operations (like transpose, reshape, etc) to find the true parent computation nodes.

        Args:
            node (Node): The node to get parents for.

        Returns:
            list[Node]: List of parent nodes, excluding reformat operations.
        """
        results = []
        for parent in get_parents(node):
            if parent.target not in cls.get_reformat_targets():
                results.append(parent)
            else:
                results.extend(cls.get_parents(parent))
        return results

    @classmethod
    def get_users(cls, node: Node) -> list[Node]:
        """Get user nodes, skipping over reformat operations.

        This method recursively traverses down the graph to find user nodes, skipping over any
        reformat operations (like transpose, reshape, etc) to find the true user computation nodes.

        Args:
            node (Node): The node to get users for.

        Returns:
            list[Node]: List of user nodes, excluding reformat operations.
        """
        results = []
        for user in node.users:
            if user.target not in cls.get_reformat_targets():
                results.append(user)
            else:
                results.extend(cls.get_users(user))
        return results


T = TypeVar("T")


def are_same_as_sets(one: list[T], another: list[T]) -> bool:
    """Check if two lists of objects consists of the same set of elements.

    Note: this is a workaround for `set(one) == set(another)`, which does not work
        if `T` is not a hashable type, for example, `torch.SymInt`.

    Args:
        one (list[T]): a list of objects
        another (list[T]): another list containing the objects of the same type.

    Returns:
        bool: True if two lists consists of the same set of elements, False otherwise.
    """
    if len(one) != len(another):
        return False
    one = one[:]
    another = another[:]
    while one:
        s = one.pop()
        matched_idx: int
        for i, t in enumerate(another):
            if s is t:
                matched_idx = i
                break
        else:
            return False
        another.pop(matched_idx)
    return True
