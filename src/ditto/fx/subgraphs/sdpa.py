# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import math
import operator
from functools import reduce

import torch
from torch._ops import OpOverload
from torch.fx import Node
from typing_extensions import Self

from ...constants import INPUT_IDS_UNSQUEEZE_DIM
from ...types import NodeCondition, Number, ShapeArg
from ..nodes import BMM, Add, ATenOp, DivScalar, MulScalar, Permute, Reshape, ScaledDotProductAttention, Softmax
from .path import Path, TrailingReformatPath
from .subgraph import Subgraph


class ScaledDotProductAttentionSubgraph(Subgraph):
    qk_bmm: BMM
    av_bmm: BMM
    external_qk_scalings: tuple[Node, ...]
    scale: Number

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (av_bmm := BMM.specialize_from(node))
            and (softmax := find_softmax(av_bmm))
            and (qk_bmm_and_internal_scale := find_qk_bmm_and_internal_scale(softmax))
        ):
            return None

        qk_bmm, internal_scale = qk_bmm_and_internal_scale

        q_external_scaling_path = AttentionScalingPath.configure_from(qk_bmm.this)
        k_external_scaling_path = AttentionScalingPath.configure_from(qk_bmm.other)
        external_qk_scalings = q_external_scaling_path.scaling_nodes + k_external_scaling_path.scaling_nodes

        scale = internal_scale * q_external_scaling_path.scale * k_external_scaling_path.scale
        return cls(qk_bmm=qk_bmm, av_bmm=av_bmm, external_qk_scalings=external_qk_scalings, scale=scale)

    @property
    def query(self) -> ATenOp:
        return ATenOp._specialize_from(self.qk_bmm.this)

    @property
    def key(self) -> ATenOp:
        return ATenOp._specialize_from(self.qk_bmm.other)

    @property
    def value(self) -> ATenOp:
        return ATenOp._specialize_from(self.av_bmm.other)

    def insert_fused_graph(self) -> Node | None:
        if not (
            (query_shape := self.query.output_shape_arg)
            and (output_tensor := self.av_bmm.output_shape_arg)
            and isinstance(head_size := query_shape[-1], int)
        ):
            return None

        if not math.isclose(self.scale, head_size ** (-0.5)):
            # TODO: Support attentions with non-default scale
            return None

        for scaling in self.external_qk_scalings:
            assert isinstance(scaling, Node)
            scaling.replace_all_uses_with(scaling.all_input_nodes[0])

        graph = self.value.node.graph
        with graph.inserting_before(self.av_bmm.node):
            q = Reshape.create(graph, self.query, get_unsqueezed_shape(query_shape))
            k = Reshape.create(graph, Permute.create(graph, self.key, [0, 2, 1]), get_unsqueezed_shape(query_shape))
            v = Reshape.create(graph, self.value, get_unsqueezed_shape(query_shape))
            sdpa = ScaledDotProductAttention.create(graph, q, k, v)
            output = Reshape.create(graph, sdpa.node, get_unsqueezed_shape(output_tensor))
        return output.node


def find_softmax(av_bmm: BMM) -> Softmax | None:
    return TrailingReformatPath.traceback(Softmax, av_bmm.this)


def find_qk_bmm_and_internal_scale(softmax: Softmax) -> tuple[BMM, Number] | None:
    post_bias_attn_path = AttentionScalingPath.configure_from(softmax.this)
    post_bias_attn_weight = post_bias_attn_path.top
    if not ((bias_add := Add.specialize_from(post_bias_attn_weight)) and len(bias_add.node.all_input_nodes) == 2):
        return None
    assert isinstance(bias_add.this, Node)

    pre_bias_attn_path = AttentionScalingPath.configure_from(
        bias_add.this, extra_whitelist=(torch.ops.aten.tanh.default,)
    )
    pre_bias_attn_weight = pre_bias_attn_path.top

    if qk_bmm := BMM.specialize_from(pre_bias_attn_weight):
        return qk_bmm, post_bias_attn_path.scale * pre_bias_attn_path.scale

    return None


def get_unsqueezed_shape(shape_arg: ShapeArg) -> ShapeArg:
    return [
        *shape_arg[:INPUT_IDS_UNSQUEEZE_DIM],
        1,
        *shape_arg[INPUT_IDS_UNSQUEEZE_DIM:],
    ]


class AttentionScalingPath(Path):
    @classmethod
    def get_reformat_targets(cls) -> tuple[OpOverload, ...]:
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
    def get_scaling_targets(cls) -> tuple[OpOverload, ...]:
        return (
            torch.ops.aten.div.Scalar,
            torch.ops.aten.mul.Scalar,
        )

    @classmethod
    def get_allowed_targets(cls) -> tuple[OpOverload, ...]:
        return cls.get_reformat_targets() + cls.get_scaling_targets()

    @classmethod
    def configure_from(
        cls,
        node: Node,
        *,
        break_if: NodeCondition = lambda _: False,
        max_len: int = 10,
        extra_whitelist: tuple[OpOverload, ...] | None = None,
    ) -> Self:
        return super().configure_from(
            node,
            break_if=lambda n: (n.target not in cls.get_allowed_targets() + (extra_whitelist or ())) or break_if(n),
            max_len=max_len,
        )

    @property
    def scaling_nodes(self) -> tuple[Node, ...]:
        return tuple(node for node in self.node_seq if node.target in self.get_scaling_targets())

    @property
    def scale(self) -> Number:
        return reduce(
            operator.mul,
            [mul.other for n in self.scaling_nodes if (mul := MulScalar.specialize_from(n))],
            reduce(
                operator.truediv, [div.other for n in self.scaling_nodes if (div := DivScalar.specialize_from(n))], 1.0
            ),
        )
