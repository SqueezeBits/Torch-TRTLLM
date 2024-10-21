from torch.fx import Node


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
