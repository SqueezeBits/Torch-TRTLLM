import logging

import torch
from torch.fx import Graph, Node
from torch.fx.passes.shape_prop import TensorMetadata, _extract_tensor_metadata

logger = logging.getLogger(__name__)


def find_closest_common_ancestor(x: Node, y: Node) -> Node | None:
    # Helper function to find all ancestors along with their depths for a given node
    def get_ancestors_with_depth(node: Node) -> dict:
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


def find_or_create_placeholder_sym_size(graph: Graph, name: str, idx: int = 0) -> Node | None:
    if name not in (placeholders := {node.name: node for node in graph.nodes if node.op == "placeholder"}):
        logger.warning(f"No such placholder: {name}")
        return None
    placeholder = placeholders[name]
    for user in placeholder.users:
        if user.target is torch.ops.aten.sym_size.int and len(user.args) == 2 and user.args[1] == idx:
            return user
    last_placeholder = [*placeholders.values()][-1]
    with graph.inserting_after(last_placeholder):
        return graph.call_function(torch.ops.aten.sym_size.int, (placeholder, 0))


def get_tensor_metadata(node: Node) -> TensorMetadata | None:
    if isinstance(tensor_meta := node.meta.get("tensor_meta"), TensorMetadata):
        return tensor_meta
    if isinstance(val := node.meta.get("val"), torch.Tensor):
        return _extract_tensor_metadata(val)
    return None


def populate_metadata(
    node: Node,
    tensor_metadata: TensorMetadata,
    shape: torch.Size | None = None,
) -> None:
    if shape is None:
        node.meta["tensor_meta"] = tensor_metadata
        return
    tensor_meta_as_dict = tensor_metadata._asdict()
    tensor_meta_as_dict["shape"] = shape
    node.meta["tensor_meta"] = TensorMetadata(**tensor_meta_as_dict)
