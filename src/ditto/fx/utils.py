from typing import Any, TypeVar, cast, overload
from weakref import WeakKeyDictionary

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Graph, Node
from torch.fx.passes.shape_prop import TensorMetadata, _extract_tensor_metadata

from ..contexts import detailed_sym_node_str


def find_closest_common_ancestor(x: Node, y: Node) -> Node | None:
    """Find the closest common ancestor node between two nodes in a graph.

    Args:
        x (Node): First node
        y (Node): Second node

    Returns:
        Node | None: The closest common ancestor node if one exists, None otherwise
    """
    # Get ancestors with their depths for both nodes x and y
    ancestors_x = get_ancestors_with_depth(x)
    ancestors_y = get_ancestors_with_depth(y)

    # Find common ancestors
    common_ancestors = set(ancestors_x.keys()).intersection(ancestors_y.keys())

    # If no common ancestor is found, return None
    if not common_ancestors:
        return None

    # Find the closest common ancestor based on minimum depth
    closest_common_ancestor = None
    min_depth = float("inf")

    for ancestor in common_ancestors:
        # The effective depth of the ancestor is the minimum of its depth in both subtrees
        depth = min(ancestors_x[ancestor], ancestors_y[ancestor])
        if depth < min_depth:
            min_depth = depth
            closest_common_ancestor = ancestor

    return closest_common_ancestor


def get_ancestors_with_depth(node: Node) -> dict[Node, int]:
    """Get all ancestor nodes of a given node with their depths.

    Args:
        node (Node): The node to get ancestors for

    Returns:
        dict[Node, int]: Dictionary mapping ancestor nodes to their depths
    """
    ancestors: dict[Node, int] = {}
    queue = [(node, 0)]  # Initialize queue with (node, depth)

    while queue:
        current, depth = queue.pop(0)
        # If node is not visited or found at a greater depth
        if current not in ancestors or depth > ancestors[current]:
            ancestors[current] = depth
            # Traverse input nodes and increase the depth by 1
            for input_node in current.all_input_nodes:
                queue.append((input_node, depth + 1))

    return ancestors


def find_closest_common_descendant(x: Node, y: Node) -> Node | None:
    """Find the closest common descendant node between two nodes in a graph.

    Args:
        x (Node): First node
        y (Node): Second node

    Returns:
        Node | None: The closest common descendant node if one exists, None otherwise
    """
    descendants_x = get_descendants_with_depth(x)
    descendants_y = get_descendants_with_depth(y)
    common_descendants = set(descendants_x.keys()).intersection(descendants_y.keys())
    if not common_descendants:
        return None
    return min(common_descendants, key=lambda d: descendants_x[d] + descendants_y[d])


def get_descendants_with_depth(node: Node) -> dict[Node, int]:
    """Get all descendant nodes of a given node with their depths.

    Args:
        node (Node): The node to get descendants for

    Returns:
        dict[Node, int]: Dictionary mapping descendant nodes to their depths
    """
    descendants: dict[Node, int] = {}
    queue = [(node, 0)]

    while queue:
        current, depth = queue.pop(0)
        if current not in descendants or depth > descendants[current]:
            descendants[current] = depth
            for user in current.users:
                queue.append((user, depth + 1))

    return descendants


def forget_all_descendant_fake_tensors(node: Node) -> None:
    """Remove fake tensor metadata from a node and all its descendants.

    Args:
        node (Node): The root node to start removing fake tensors from
    """
    _ = node.meta.pop("val", None)
    for n in get_descendants_with_depth(node):
        _ = n.meta.pop("val", None)


def get_tensor_metadata(node: Node) -> TensorMetadata | None:
    """Get tensor metadata from a node if available.

    Args:
        node (Node): The node to get tensor metadata from

    Returns:
        TensorMetadata | None: The tensor metadata if available, None otherwise
    """
    if isinstance(val := node.meta.get("val"), torch.Tensor):
        return _extract_tensor_metadata(val)
    if isinstance(val, torch.SymInt) or node.target is torch.ops.aten.sym_size.int:
        return TensorMetadata(
            shape=torch.Size(()),
            dtype=torch.int64,
            requires_grad=False,
            stride=(),
            memory_format=None,
            is_quantized=False,
            qparams={},
        )
    return None


T = TypeVar("T", torch.Tensor, FakeTensor, torch.SymInt)


@overload
def get_val(node: Node, expected_type: type[torch.SymInt]) -> torch.SymInt | None:
    ...


@overload
def get_val(node: Node, expected_type: type[torch.Tensor]) -> FakeTensor | None:
    ...


@overload
def get_val(node: Node) -> torch.Tensor | torch.SymInt | None:
    ...


def get_val(node: Node, expected_type: type[T] | None = None) -> T | None:
    """Get the value stored in a node's metadata.

    Args:
        node (Node): The node to get the value from
        expected_type (type[T] | None, optional): Expected type of the value. Defaults to None

    Returns:
        T | None: The value if it exists and matches the expected type, None otherwise
    """
    val = node.meta.get("val")
    if expected_type is not None:
        if isinstance(val, expected_type):
            return val
        return None
    if isinstance(val, torch.Tensor | torch.SymInt):
        return cast(T, val)
    return None


def get_fake_mode(graph: Graph) -> FakeTensorMode | None:
    """Get the fake tensor mode from a graph if one exists.

    Args:
        graph (Graph): The graph to get the fake tensor mode from

    Returns:
        FakeTensorMode | None: The fake tensor mode if one exists, None otherwise
    """
    for node in graph.nodes:
        if isinstance(val := node.meta.get("val"), FakeTensor):
            return val.fake_mode
    return None


def find_sym_size_node(graph: Graph, s: torch.SymInt) -> Node:
    """Find the node that produces a given symbolic integer in a graph.

    Args:
        graph (Graph): The graph to search in
        s (torch.SymInt): The symbolic integer to find

    Returns:
        Node: The node that produces the symbolic integer

    Raises:
        RuntimeError: If no node is found that produces the symbolic integer
    """
    cache: WeakKeyDictionary[Node, torch.SymInt] | None = None
    if graph_module := graph.owning_module:
        if "symint_cache" not in graph_module.meta:
            graph_module.meta["symint_cache"] = WeakKeyDictionary()
        cache = graph_module.meta["symint_cache"]

    if cache is not None:
        for node, sym_int in cache.items():
            if sym_int is s:
                return node

    for node in graph.nodes:
        if isinstance(val := node.meta.get("val"), torch.SymInt) and val == s:
            if cache is not None:
                cache[node] = s
            return node

    with detailed_sym_node_str():
        raise RuntimeError(f"Failed to find a node producing the symbolic integer {s}")


def replace_all_uses_with(s: Node, replace_with: Node, exclude_nodes: list[Node] | None = None) -> Node:
    """Replace all uses of the source node with the target node.

    This function replaces the source node with the target node wherever it is used.
    It preserves the target node's original arguments to prevent self-referencing issues.

    Args:
        s (Node): The source node to be replaced
        replace_with (Node): The target node to replace the source node
        exclude_nodes (list[Node] | None, optional): Nodes to exclude from the replacement. Defaults to None

    Returns:
        Node: The target node
    """
    preserved_nodes: dict[Node, tuple[Any, ...]] = {}
    if exclude_nodes is not None:
        for node in exclude_nodes:
            preserved_nodes[node] = (node.args, node.kwargs)
    for node in s.graph.nodes:
        flat_inputs, _ = pytree.tree_flatten((node.args, node.kwargs))
        if s in flat_inputs and node is replace_with:
            preserved_nodes[node] = (node.args, node.kwargs)

    s.replace_all_uses_with(replace_with)
    for node, (args, kwargs) in preserved_nodes.items():
        node.args, node.kwargs = args, kwargs
