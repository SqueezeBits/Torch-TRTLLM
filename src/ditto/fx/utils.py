from weakref import WeakKeyDictionary

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Graph, Node
from torch.fx.passes.shape_prop import TensorMetadata, _extract_tensor_metadata

from ..contexts import detailed_sym_node_str


def find_closest_common_ancestor(x: Node, y: Node) -> Node | None:
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
    ancestors: dict[Node, int] = {}
    stack = [(node, 0)]  # Initialize stack with (node, depth)

    while stack:
        current, depth = stack.pop()
        # If node is not visited or found at a greater depth
        if current not in ancestors or depth < ancestors[current]:
            ancestors[current] = depth
            # Traverse input nodes and increase the depth by 1
            for input_node in current.all_input_nodes:
                stack.append((input_node, depth + 1))

    return ancestors


def forget_all_descendant_fake_tensors(node: Node) -> None:
    nodes = [node]
    while nodes:
        n = nodes.pop()
        _ = n.meta.pop("val", None)
        nodes.extend(n.users)


def get_tensor_metadata(node: Node) -> TensorMetadata | None:
    if isinstance(tensor_meta := node.meta.get("tensor_meta"), TensorMetadata):
        return tensor_meta
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


def get_fake_mode(graph: Graph) -> FakeTensorMode | None:
    for node in graph.nodes:
        if isinstance(val := node.meta.get("val"), FakeTensor):
            return val.fake_mode
    return None


def find_sym_size_node(graph: Graph, s: torch.SymInt) -> Node:
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
