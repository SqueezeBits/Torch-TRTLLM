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

from typing import ClassVar, overload

import torch
from torch.fx import Node
from typing_extensions import Self

from ...types import SymbolicInteger
from ..nodes import IndexPut, MulTensorTensor, SelectInt, Softmax, ToCopy
from ..targets import AllReducePlugin
from ..utils import find_nearest, get_tensor_metadata
from .gated_mlp import GatedMLP
from .linear import Linear, MMType
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
    up_proj: Linear
    gate_proj: Linear
    down_proj: Linear
    final_hidden_states: Node

    UNUSED_TARGETS: ClassVar[tuple] = (torch.ops.aten.sym_constrain_range_for_size.default,)

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if not (
            (select := SelectInt.specialize_from(node))
            and (gated_mlp := find_nearest(GatedMLP, select.node, follow_parent=False, follow_first_only=False))
            and (gated_mlp_mul := gated_mlp.mul)
            and (down_proj := find_nearest(Linear, gated_mlp_mul.node, follow_parent=False))
            and (len(users := list(down_proj.output_node.users)) == 1)
            and (
                expert_hidden_states := (
                    list(users[0].users)[0] if isinstance(users[0].target, AllReducePlugin) else users[0]
                )
            )
            and (weighted_expert := MulTensorTensor.specialize_from(expert_hidden_states))
            and (len(users := list(weighted_expert.users)) == 1)
            and (final_hidden_states := IndexPut.specialize_from(users[0]))
        ):
            return None

        return cls(
            index=select.index,
            entry_node=select,
            up_proj=gated_mlp.up_proj,
            gate_proj=gated_mlp.gate_proj,
            down_proj=down_proj,
            final_hidden_states=final_hidden_states.node,
        )

    # pylint: disable=duplicate-code
    def find_unused_nodes(self) -> set[Node]:
        """Find unused nodes in the expert subgraph.

        This method performs a breadth-first search starting from the expert's entry node
        to find nodes that are unused (have no users) and match the UNUSED_TARGETS. This is
        used to identify nodes that can be removed.

        Returns:
            set[Node]: A set of nodes that are unused in the expert subgraph.
        """
        visited = set()
        queue = [self.entry_node.node]
        unused_nodes = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current.target in self.UNUSED_TARGETS and len(current.users) == 0:
                unused_nodes.add(current)
            for user in current.users:
                if user not in visited:
                    queue.append(user)
        return unused_nodes


# pylint: disable=too-many-locals
class MoESubgraph(Subgraph):
    """A Mixture of Experts (MoE) layer subgraph.

    This subgraph identifies the pattern of operations used for a Mixture of Experts (MoE) layer.
    It includes:
    - Routing logics for expert selection
    - Expert computations and aggregation
    - Optional shared experts

    Attributes:
        hidden_states (Node): The input hidden states node.
        router (Linear): The router linear layer for expert selection.
        router_logits (Node): The router logits node before softmax.
        top_k (int): Number of experts to select per token.
        experts (list[Expert]): List of Expert objects representing each expert network.
        shared_experts (list[tuple[Linear, Linear, Linear]]): List of shared expert weight tuples.
            Each tuple contains (up_proj, gate_proj, down_proj) Linear layers.
        final_hidden_states (Node): The final output hidden states after MoE computation.
        unused_nodes (set[Node]): Set of unused nodes in the MoE computation graph.
    """

    hidden_states: Node
    router: Linear
    router_logits: Node
    top_k: int
    experts: list[Expert]
    shared_experts: list[tuple[Linear, Linear, Linear]]
    final_hidden_states: Node
    unused_nodes: set[Node]

    @property
    def number_of_experts(self) -> int:
        """Get the number of experts in the MoE layer.

        Returns:
            int: The number of expert networks
        """
        return len(self.experts)

    @property
    def expert_hidden_size(self) -> int:
        """Get the hidden dimension size used by each expert.

        Returns:
            int: Hidden dimension size for expert networks
        """
        assert (
            expert_weight := get_tensor_metadata(self.extract_weights(self.experts[0])[0])
        ) is not None, "expert weight's metadata is not found"
        return expert_weight.shape[0]

    @property
    def expert_inter_size(self) -> int:
        """Get the intermediate dimension size used by each expert.

        Returns:
            int: Intermediate dimension size for expert networks
        """
        assert (
            expert_weight := get_tensor_metadata(self.extract_weights(self.experts[0])[0])
        ) is not None, "expert weight's metadata is not found"
        return expert_weight.shape[1]

    @property
    def shared_expert_intermediate_size(self) -> int:
        """Get the intermediate dimension size used by the shared expert.

        Returns:
            int: Intermediate dimension size for the shared expert network
        """
        if len(self.shared_experts) == 0:
            return 0
        assert (
            shared_expert_weight := get_tensor_metadata(self.extract_weights(self.shared_experts[0])[0])
        ) is not None, "shared expert weight's metadata is not found"
        return shared_expert_weight.shape[1]

    @property
    def router_logits_dtype(self) -> torch.dtype:
        """Get the data type of the router logits.

        Returns:
            torch.dtype: The data type of the router logits tensor
        """
        return self.router_logits.meta["val"].dtype

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        # The common pattern for Mixture of Experts (MoE) is:
        # router -> softmax -> top-k -> one-hot
        # This pattern is used in Qwen and Mixtral.
        if not (
            (one_hot := OneHot.configure_from(node))
            and (get_item := TrailingReformatPath.configure_from(one_hot.this).top)
            and (topk := TopK.configure_from(get_item))
            and (softmax := find_nearest(Softmax, topk.this, follow_first_only=False))
            and (router := TrailingReformatPath.traceback(Linear, softmax.this))
        ):
            return None

        if not (expert := find_nearest(Expert, one_hot.eq.node, follow_parent=False)):
            raise NotImplementedError(f"Unsupported expert graph found from: {one_hot.eq.node}")

        experts: list[Expert] = []
        unused_nodes = set()
        for expert_entry in expert.entry_node.this.users:
            if not (expert := Expert.configure_from(expert_entry)):
                raise NotImplementedError(f"Unsupported expert graph found from: {expert_entry}")
            experts.append(expert)
            unused_nodes.update(expert.find_unused_nodes())

        expert_hidden_states: ToCopy | MMType
        if to_copy := ToCopy.specialize_from(router.mm.this):
            # When the router is casted to FP32.
            expert_hidden_states = to_copy
        else:
            expert_hidden_states = router.mm

        shared_experts: list[tuple[Linear, Linear, Linear]] = []
        if len(TrailingReformatPath.get_parents(expert_hidden_states.this)) != 1:
            raise NotImplementedError(f"Unsupported shared expert graph found from: {expert_hidden_states.this}")
        common_hidden_states = TrailingReformatPath.get_parents(expert_hidden_states.this)[0]
        for user in TrailingReformatPath.get_users(common_hidden_states):
            if user == router.mm.node or len(user.all_input_nodes[0].users) == len(experts):
                continue
            if not (
                Linear.configure_from(user)
                and (gated_mlp := find_nearest(GatedMLP, user, follow_parent=False))
                and (gated_mlp_mul := gated_mlp.mul)
                and (down_proj := find_nearest(Linear, gated_mlp_mul.node, follow_parent=False))
            ):
                continue
            if len(shared_experts) > 0 and down_proj in [down for _, _, down in shared_experts]:
                continue
            for linear in (linears := (gated_mlp.up_proj, gated_mlp.gate_proj, down_proj)):
                linear.mark_linear_type_as("shared_expert")
            shared_experts.append(linears)

        if Softmax.specialize_from(topk.this):
            # greedy
            router_logits = softmax.this
        else:
            # group_limited_greedy
            # This topk method is used in DeepSeek-V2. Deepseek-V2-Lite uses greedy method.
            #
            # Currently, it only finds proper router logits and pattern matching
            # for full group_limited_greedy graph is not implemented, as it's not necessary.
            router_logits = topk.this
        experts.sort(key=lambda expert: expert.index)
        final_hidden_states = experts[-1].final_hidden_states
        return cls(
            hidden_states=expert_hidden_states.this,
            router=router,
            router_logits=router_logits,
            top_k=int(topk.k),
            experts=experts,
            shared_experts=shared_experts,
            final_hidden_states=final_hidden_states,
            unused_nodes=unused_nodes,
        )

    @staticmethod
    @overload
    def extract_weights(expert: Expert) -> tuple[Node, Node, Node]:
        ...

    @staticmethod
    @overload
    def extract_weights(expert: tuple[Linear, Linear, Linear]) -> tuple[Node, Node, Node]:
        ...

    @staticmethod
    def extract_weights(expert: Expert | tuple[Linear, Linear, Linear]) -> tuple[Node, Node, Node]:
        """Extract weight nodes from an expert or tuple of linear layers.

        Args:
            expert: Either an Expert instance or a tuple of (up_proj, gate_proj, down_proj) Linear layers

        Returns:
            A tuple of (up_proj_weight, gate_proj_weight, down_proj_weight) nodes
        """
        if isinstance(expert, Expert):
            return (expert.up_proj.mm.other, expert.gate_proj.mm.other, expert.down_proj.mm.other)
        return (expert[0].mm.other, expert[1].mm.other, expert[2].mm.other)
