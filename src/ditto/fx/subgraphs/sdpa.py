from torch.fx import Node
from typing_extensions import Self

from ...constants import INPUT_IDS_UNSQUEEZE_DIM
from ...types import ShapeArg
from ..nodes import BMM, Add, ATenOp, Div, Permute, Reshape, ScaledDotProductAttention, Softmax
from .path import TrailingReformatPath
from .subgraph import Subgraph


class ScaledDotProductAttentionSubgraph(Subgraph):
    qk_bmm: BMM
    kv_bmm: BMM

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (kv_bmm := BMM.specialize_from(node))
            and (softmax := TrailingReformatPath.traceback(Softmax, kv_bmm.this))
            and (attn_weight_div := traceback_attn_weight_div(softmax))
            and isinstance(qk_t := attn_weight_div.this, Node)
            and (qk_bmm := TrailingReformatPath.traceback(BMM, qk_t))
        ):
            return None
        return cls(qk_bmm=qk_bmm, kv_bmm=kv_bmm)

    @property
    def query(self) -> ATenOp:
        return ATenOp._specialize_from(self.qk_bmm.this)

    @property
    def key(self) -> ATenOp:
        return ATenOp._specialize_from(self.qk_bmm.other)

    @property
    def value(self) -> ATenOp:
        return ATenOp._specialize_from(self.kv_bmm.other)

    def insert_fused_graph(self) -> Node | None:
        if not (
            (query_shape := self.query.output_shape_arg)
            and (key_shape := self.key.output_shape_arg)
            and (value_shape := self.value.output_shape_arg)
            and (output_tensor := self.kv_bmm.output_shape_arg)
        ):
            return None

        graph = self.value.node.graph
        with graph.inserting_before(self.kv_bmm.node):
            q = Reshape.create(graph, self.query, get_unsqueezed_shape(query_shape))
            k = Reshape.create(graph, Permute.create(graph, self.key, [0, 2, 1]), get_unsqueezed_shape(key_shape))
            v = Reshape.create(graph, self.value, get_unsqueezed_shape(value_shape))
            sdpa = ScaledDotProductAttention.create(graph, q, k, v)
            output = Reshape.create(graph, sdpa.node, get_unsqueezed_shape(output_tensor))
        return output.node


def traceback_attn_weight_div(softmax: Softmax) -> Div | None:
    if attn_mask_add := TrailingReformatPath.traceback(Add, softmax.this):
        for operand in (attn_mask_add.this, attn_mask_add.other):
            if isinstance(operand, Node) and (div := TrailingReformatPath.traceback(Div, operand)):
                return div
        return None

    return TrailingReformatPath.traceback(Div, softmax.this)


def get_unsqueezed_shape(shape_arg: ShapeArg) -> ShapeArg:
    return [
        *shape_arg[:INPUT_IDS_UNSQUEEZE_DIM],
        1,
        *shape_arg[INPUT_IDS_UNSQUEEZE_DIM:],
    ]
