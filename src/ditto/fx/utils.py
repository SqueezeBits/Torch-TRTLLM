import torch
from loguru import logger
from torch.fx import Graph, Node
from torch.fx.passes.shape_prop import TensorMetadata, _extract_tensor_metadata


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


def find_or_create_placeholder_sym_size(graph: Graph, name: str, dim: int = 0) -> Node | None:
    if name not in (placeholders := {node.name: node for node in graph.nodes if node.op == "placeholder"}):
        logger.warning(f"No such placholder: {name}")
        return None
    placeholder = placeholders[name]
    for user in placeholder.users:
        if user.target is torch.ops.aten.sym_size.int and len(user.args) == 2 and user.args[1] == dim:
            return user
    last_placeholder = [*placeholders.values()][-1]
    with graph.inserting_after(last_placeholder):
        node = graph.call_function(torch.ops.aten.sym_size.int, (placeholder, dim))
        if (metadata := get_tensor_metadata(placeholder)) and isinstance(s := metadata.shape[dim], torch.SymInt):
            node.meta["val"] = s
        return node


def get_tensor_metadata(node: Node) -> TensorMetadata | None:
    if isinstance(tensor_meta := node.meta.get("tensor_meta"), TensorMetadata):
        return tensor_meta
    if isinstance(val := node.meta.get("val"), torch.Tensor):
        return _extract_tensor_metadata(val)
    return None


def populate_tensor_metadata(
    node: Node,
    tensor_metadata: TensorMetadata | torch.Tensor,
    *,
    shape: torch.Size | tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> TensorMetadata:
    if isinstance(tensor_metadata, torch.Tensor):
        tensor_metadata = _extract_tensor_metadata(tensor_metadata)
    tensor_meta_as_dict = tensor_metadata._asdict()
    if shape is not None:
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        tensor_meta_as_dict["shape"] = shape
    if dtype is not None:
        tensor_meta_as_dict["dtype"] = dtype
    node.meta["tensor_meta"] = (metadata := TensorMetadata(**tensor_meta_as_dict))
    return metadata


def traceback_reformats(node: Node) -> Node:
    if node.target not in (
        torch.ops.aten.clone.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.unsqueeze.default,
    ):
        return node
    return traceback_reformats(node.all_input_nodes[0])
