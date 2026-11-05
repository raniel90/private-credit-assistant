"""Shared identity and embedding types for the local first-party RAG lab."""

from dataclasses import dataclass

from fastembed import TextEmbedding


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
COLLECTION = "research_chunks"


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    groups: frozenset[str]


class FastEmbedder:
    def __init__(self) -> None:
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]
