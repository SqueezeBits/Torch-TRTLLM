from torch.fx import Node
from typing_extensions import Self

from ...constants import INPUT_IDS_UNSQUEEZE_DIM
from ...types import ShapeArg, Number
from ..nodes import Add, BMM, ATenOp, Permute, Reshape, ScaledDotProductAttention, Softmax
from .subgraph import Subgraph
from .scaling_reformat_path import ScalingReformatPath


class ScaledDotProductAttentionSubgraph(Subgraph):
    qk_bmm: BMM
    kv_bmm: BMM

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (kv_bmm := BMM.specialize_from(node))
            and (softmax := find_softmax(kv_bmm))
            and (qk_bmm := find_qk_bmm(softmax))
        ):
            return None
        return cls(qk_bmm=qk_bmm, kv_bmm=kv_bmm)

    @property
    def query(self) -> ATenOp:
        return ATenOp._specialize_from(self.qk_bmm.this)

    @property
    def key(self) -> ATenOp:
        key_path = ScalingReformatPath.configure_from(self.qk_bmm.other, max_len=3)
        permute = Permute.specialize_from(key_path.top)
        assert permute
        return ATenOp._specialize_from(permute.this)

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

        q_scale = pop_qk_scaling_from_graph(self.qk_bmm.this)
        k_scale = pop_qk_scaling_from_graph(self.qk_bmm.other)
        assert isinstance(head_dim := query_shape[-1], int)
        if not q_scale*k_scale == head_dim**(-1/2):
            raise NotImplementedError("wtf??")

        graph = self.value.node.graph
        with graph.inserting_before(self.kv_bmm.node):
            q = Reshape.create(graph, self.query, get_unsqueezed_shape(query_shape))
            k = Reshape.create(graph, self.key, key_shape)
            v = Reshape.create(graph, self.value, get_unsqueezed_shape(value_shape))
            sdpa = ScaledDotProductAttention.create(graph, q, k, v)
            output = Reshape.create(graph, sdpa.node, get_unsqueezed_shape(output_tensor))
        return output.node


def from_sdpa(node: Node) -> bool:
    return "torch.nn.functional.scaled_dot_product_attention" in node.meta.get("stack_trace", "")


def find_softmax(kv_bmm: BMM) -> Softmax | None:
    attn_weight = kv_bmm.this
    if path := ScalingReformatPath.configure_from(attn_weight):
        attn_weight = path.top
    return Softmax.specialize_from(attn_weight)


def find_qk_bmm(softmax: Softmax) -> BMM | None:
    attn_weight = ScalingReformatPath.configure_from(softmax.this).top
    bias_add = Add.specialize_from(attn_weight)
    if bias_add is None or not isinstance(bias_add.this, Node):
        return None
    return BMM.specialize_from(ScalingReformatPath.configure_from(bias_add.this).top)


def pop_qk_scaling_from_graph(q_or_k: Node) -> Number:
    if path := ScalingReformatPath.configure_from(q_or_k, max_len=3):
        scale = path.scale
        for scaling_node in path.scalings:
            parent = scaling_node.all_input_nodes[0]
            assert len(scaling_node.users) == 1
            scaling_node.replace_all_uses_with(parent)
        return scale
    return 1.0


def get_unsqueezed_shape(shape_arg: ShapeArg) -> ShapeArg:
    return [
        *shape_arg[:INPUT_IDS_UNSQUEEZE_DIM],
        1,
        *shape_arg[INPUT_IDS_UNSQUEEZE_DIM:],
    ]
