import operator

from torch.fx import Node

from ..nodes import Cat, Split
from .node_wise_pass import NodeWiseOptimizationPass


class FuseConsecutiveSplitConcat(NodeWiseOptimizationPass):
    """Fuse consecutive split and concat that is identical to nop."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            (split := Split.specialize_from(node))
            and (
                getitems := [
                    user for user in node.users if (user.op == "call_function" and user.target is operator.getitem)
                ]
            )
            and (
                (isinstance(split.split_size, int) and len(getitems) == split.split_size)
                or (isinstance(split.split_size, list) and len(getitems) == len(split.split_size))
            )
        ):
            return {}
        cat_nodes = {
            cat.node
            for getitem in getitems
            for user in getitem.users
            if (cat := Cat.specialize_from(user)) and cat.tensors == getitems and cat.dim == split.dim
        }
        return {cat_node: split.this for cat_node in cat_nodes}
