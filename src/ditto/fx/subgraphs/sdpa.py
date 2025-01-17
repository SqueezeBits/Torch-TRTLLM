# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false

import math
import operator
from functools import reduce

import torch
from torch._ops import OpOverload
from torch.fx import Node
from typing_extensions import Self

from ...constants import INPUT_IDS_UNSQUEEZE_DIM
from ...types import NodeCriterion, Number, ShapeArg
from ..nodes import BMM, Add, ATenOp, DivScalar, MulScalar, Permute, Reshape, ScaledDotProductAttention, Softmax
from .path import Path, TrailingReformatPath
from .subgraph import Subgraph


class ScaledDotProductAttentionSubgraph(Subgraph):
    """A subgraph representing a scaled dot-product attention operation.

    This subgraph identifies the components of a scaled dot-product attention, including
    query-key and attention-value matrix multiplications, scaling factors, and external
    scaling nodes. It provides methods to configure the subgraph and fuse it into a single
    graph representation.

    Attributes:
        qk_bmm (BMM): The batch matrix multiplication for the query-key operation.
        av_bmm (BMM): The batch matrix multiplication for the attention-value operation.
        external_qk_scalings (tuple[Node, ...]): Nodes responsible for external scaling of
            the query-key scores.
        scale (Number): The combined scaling factor for the attention weight before softmax operation.
    """

    qk_bmm: BMM
    av_bmm: BMM
    external_qk_scalings: tuple[Node, ...]
    scale: Number

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        """Configure a scaled dot-product attention subgraph from a given node.

        Identifies the scaled dot-product attention components, including the
        query-key BMM, attention-value BMM, and scaling factors. It combines these
        components into a unified subgraph.

        Args:
            node (Node): The starting node for configuration.

        Returns:
            ScaledDotProductAttentionSubgraph | None: The configured subgraph, or None
                if the node does not represent a valid attention pattern.
        """
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
        """The query tensor operation derived from the query-key BMM."""
        return ATenOp._specialize_from(self.qk_bmm.this)

    @property
    def key(self) -> ATenOp:
        """The key tensor operation derived from the query-key BMM."""
        return ATenOp._specialize_from(self.qk_bmm.other)

    @property
    def value(self) -> ATenOp:
        """The value tensor operation derived from the attention-value BMM."""
        return ATenOp._specialize_from(self.av_bmm.other)

    def insert_fused_graph(self) -> Node | None:
        """Insert a fused graph representation of the scaled dot-product attention.

        Fuses the attention subgraph into a single scaled_dot_product_attention function call that
        represents the entire scaled dot-product attention operation.

        Returns:
            Node | None: The resulting fused graph node, or None if the fusion is not
                feasible (e.g., due to unsupported scaling factors).
        """
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
    """Find the softmax operation in the trailing path of an attention-value BMM.

    Args:
        av_bmm (BMM): The attention-value batch matrix multiplication.

    Returns:
        Softmax | None: The softmax operation if found, or None otherwise.
    """
    return TrailingReformatPath.traceback(Softmax, av_bmm.this)


def find_qk_bmm_and_internal_scale(softmax: Softmax) -> tuple[BMM, Number] | None:
    """Find the query-key BMM and internal scaling factor for a softmax operation.

    Args:
        softmax (Softmax): The softmax operation for which to find the query-key BMM
            and scaling factor.

    Returns:
        tuple[BMM, Number] | None: A tuple containing the query-key BMM and the combined
            scaling factor, or None if no valid components are found.
    """
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
    """Compute a new shape by unsqueezing a dimension.

    Inserts a singleton dimension into the given shape argument at the position
    defined by INPUT_IDS_UNSQUEEZE_DIM.

    Args:
        shape_arg (ShapeArg): The original shape argument.

    Returns:
        ShapeArg: The modified shape argument with an unsqueezed dimension.
    """
    return [
        *shape_arg[:INPUT_IDS_UNSQUEEZE_DIM],
        1,
        *shape_arg[INPUT_IDS_UNSQUEEZE_DIM:],
    ]


class AttentionScalingPath(Path):
    """A path representation for tracking scaling operations in attention computation.

    This class provides utilities to identify and manage scaling and reformatting
    operations along a computational path. It supports configuring paths with custom
    criteria and extracting scaling-related nodes.
    """

    @classmethod
    def get_scaling_targets(cls) -> tuple[OpOverload, ...]:
        """Get the operations considered as scaling targets.

        Scaling targets are operations that perform scalar division or multiplication.

        Returns:
            tuple[OpOverload, ...]: A tuple of operations such as `torch.ops.aten.div.Scalar`
                and `torch.ops.aten.mul.Scalar`.
        """
        return (
            torch.ops.aten.div.Scalar,
            torch.ops.aten.mul.Scalar,
        )

    @classmethod
    def get_allowed_targets(cls) -> tuple[OpOverload, ...]:
        """Get all operations allowed in the scaling path.

        Combines reformat targets (from `TrailingReformatPath`) with scaling targets
        defined in this class.

        Returns:
            tuple[OpOverload, ...]: A tuple of operations allowed in the path.
        """
        return TrailingReformatPath.get_reformat_targets() + cls.get_scaling_targets()

    @classmethod
    def configure_from(
        cls,
        node: Node,
        *,
        break_if: NodeCriterion = lambda _: False,
        max_len: int = 10,
        extra_whitelist: tuple[OpOverload, ...] | None = None,
    ) -> Self:
        """Configure a scaling path starting from a specific node.

        Traverses the computational graph upward from the specified node to construct
        a path that includes scaling and reformatting operations. Traversal stops when
        a break condition is met or the maximum path length is reached.

        Args:
            node (Node): The starting node for the path.
            break_if (NodeCriterion, optional): A function to determine when to stop
                traversal. Defaults to a lambda that always returns False.
            max_len (int, optional): The maximum length of the path. Defaults to 10.
            extra_whitelist (tuple[OpOverload, ...] | None, optional): Additional
                operations to allow in the path. Defaults to None.

        Returns:
            AttentionScalingPath: The configured scaling path object.
        """
        return super().configure_from(
            node,
            break_if=lambda n: (n.target not in cls.get_allowed_targets() + (extra_whitelist or ())) or break_if(n),
            max_len=max_len,
        )

    @property
    def scaling_nodes(self) -> tuple[Node, ...]:
        """The nodes in the path that perform scaling operations."""
        return tuple(node for node in self.node_seq if node.target in self.get_scaling_targets())

    @property
    def scale(self) -> Number:
        """The combined scaling factor derived from the path's scaling nodes."""
        return reduce(
            operator.mul,
            [mul.other for n in self.scaling_nodes if (mul := MulScalar.specialize_from(n))],
            reduce(
                operator.truediv, [div.other for n in self.scaling_nodes if (div := DivScalar.specialize_from(n))], 1.0
            ),
        )
