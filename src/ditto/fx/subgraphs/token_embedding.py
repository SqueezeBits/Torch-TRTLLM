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

from ..nodes import Embedding, GetAttr, Placeholder
from .subgraph import Subgraph


class TokenEmbedding(Subgraph):
    """The token embedding subgraph.

    Args:
        embedding (Embedding): The embedding node.
        input_ids (Placeholder): The input ids placeholder.
        weights (GetAttr): The weights attribute.
    """

    embedding: Embedding
    input_ids: Placeholder
    weights: GetAttr

    @classmethod
    def configure_from(cls, node: Node) -> Self | None:
        if (
            (embedding := Embedding.specialize_from(node))
            and (weights := GetAttr.specialize_from(embedding.weight))
            and weights.parameter.ndim == 2
            and (input_ids := Placeholder.specialize_from(embedding.indices))
        ):
            return cls(embedding=embedding, input_ids=input_ids, weights=weights)
        return None

    @property
    def vocab_size(self) -> int:
        """The vocabulary size of the token embedding.

        Returns:
            int: The vocabulary size of the token embedding.
        """
        return self.weights.parameter.shape[0]

    @property
    def hidden_size(self) -> int:
        """The hidden size of the token embedding.

        Returns:
            int: The hidden size of the token embedding.
        """
        return self.weights.parameter.shape[1]
