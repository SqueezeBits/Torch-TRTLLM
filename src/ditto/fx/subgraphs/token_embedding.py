from torch.fx import Node
from typing_extensions import Self

from ..nodes import Embedding, GetAttr, Placeholder
from .subgraph import Subgraph


class TokenEmbedding(Subgraph):
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
        return self.weights.parameter.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.weights.parameter.shape[1]
