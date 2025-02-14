# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch.fx import Node
from typing_extensions import Self

from ...types import SymbolicInteger
from ..nodes import IndexPut, MulTensorTensor, SelectInt, Softmax
from .gated_mlp import GatedMLP
from .linear import Linear
from .one_hot import OneHot
from .path import TrailingReformatPath
from .subgraph import Subgraph
from .topk import TopK


class Expert(Subgraph):
    """A single expert in a Mixture of Experts (MoE) subgraph.

    Attributes:
        index (int | SymbolicInteger): The index of this expert in the MoE layer.
        up_proj (Node): The up projection linear layer node.
        gate_proj (Node): The gate projection linear layer node.
        down_proj (Node): The down projection linear layer node.
        final_hidden_states (Node): The final hidden states node after expert computation.
    """

    index: int | SymbolicInteger
    entry_node: SelectInt
    up_proj: Node
    gate_proj: Node
    down_proj: Node
    final_hidden_states: Node

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (select := SelectInt.specialize_from(node))
            and (gated_mlp := GatedMLP.find_nearest(select.node, follow_parent=False, follow_first_only=False))
            and (down_proj := Linear.find_nearest(gated_mlp.mul.node, follow_parent=False))
            and (len(users := list(down_proj.output_node.users)) == 1)
            and (weighted_expert := MulTensorTensor.specialize_from(users[0]))
            and (len(users := list(weighted_expert.users)) == 1)
            and (final_hidden_states := IndexPut.specialize_from(users[0]))
        ):
            return None

        return cls(
            index=select.index,
            entry_node=select,
            up_proj=gated_mlp.up_proj.mm.node,
            gate_proj=gated_mlp.gate_proj.mm.node,
            down_proj=down_proj.mm.node,
            final_hidden_states=final_hidden_states.node,
        )


class MoESubgraph(Subgraph):
    """A Mixture of Experts (MoE) layer subgraph.

    This subgraph identifies the pattern of operations used for a Mixture of Experts (MoE) layer.
    It includes:
    - Routing logics for expert selection
    - Expert computations and aggregation

    Attributes:
        hidden_states (Node): The input hidden states node.
        router_logits (Node): The router logits node before softmax.
        experts (list[Expert]): List of Expert objects representing each expert in the MoE layer.
        final_hidden_states (Node): The final output hidden states node after MoE computation.
    """

    hidden_states: Node
    router_logits: Node
    expert_weights: list[tuple[Node, Node, Node]]
    final_hidden_states: Node

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        # The common pattern for Mixture of Experts (MoE) is:
        # softmax -> top-k -> one-hot
        # This pattern is used in Qwen, Mixtral, and DeepSeek-v1.
        if not (
            (one_hot := OneHot.configure_from(node))
            and (get_item := TrailingReformatPath.configure_from(one_hot.this).top)
            and (topk := TopK.configure_from(get_item))
            and (softmax := Softmax.specialize_from(topk.this))
            and (gate := TrailingReformatPath.traceback(Linear, softmax.this))
        ):
            return None

        if not (expert := Expert.find_nearest(one_hot.eq.node, follow_parent=False)):
            raise NotImplementedError(f"Unsupported expert graph found from: {one_hot.eq.node}")

        experts = []
        for expert_entry in expert.entry_node.this.users:
            if not (expert := Expert.configure_from(expert_entry)):
                raise NotImplementedError(f"Unsupported expert graph found from: {expert_entry}")
            experts.append(expert)

        hidden_states = TrailingReformatPath.configure_from(gate.mm.this).top
        router_logits = softmax.this
        experts.sort(key=lambda expert: expert.index)
        final_hidden_states = experts[-1].final_hidden_states
        return cls(
            hidden_states=hidden_states,
            router_logits=router_logits,
            expert_weights=cls.extract_weights(experts),
            final_hidden_states=final_hidden_states,
        )

    @staticmethod
    def extract_weights(experts: list[Expert]) -> list[tuple[Node, Node, Node]]:
        """Extract weights from a list of experts to exclude other components for MoESubgraph instances.

        Args:
            experts (list[Expert]): List of Expert objects to extract weights from.

        Returns:
            list[tuple[Node, Node, Node]]: List of tuples containing (up_proj, gate_proj, down_proj)
                weight nodes for each expert.
        """
        return [(expert.up_proj, expert.gate_proj, expert.down_proj) for expert in experts]
