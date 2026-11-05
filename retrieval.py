"""Qdrant retrieval shared by the API and MCP adapters."""

import os
from functools import lru_cache

from qdrant_client import QdrantClient, models

from core import COLLECTION, FastEmbedder, Principal


@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(url=os.getenv("RAG_LAB_QDRANT_URL", "http://127.0.0.1:6333"))


@lru_cache(maxsize=1)
def get_embedder():
    # Importing the API or running isolated tests must not download model weights.
    return FastEmbedder()


def search_documents(question: str, user: Principal, limit: int = 3) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20")
    if not user.groups:
        return []

    vector = get_embedder().embed([question])[0]
    response = get_client().query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=user.tenant_id),
                ),
                models.FieldCondition(
                    key="allowed_groups",
                    match=models.MatchAny(any=sorted(user.groups)),
                ),
            ]
        ),
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "document_id": point.payload["document_id"],
            "title": point.payload["title"],
            "text": point.payload["text"],
            "source_path": point.payload["source_path"],
            "chunk_index": point.payload["chunk_index"],
            "sha256": point.payload.get("sha256"),
            "pipeline_version": point.payload.get("pipeline_version"),
            "score": point.score,
        }
        for point in response.points
    ]


if __name__ == "__main__":
    alice = Principal("alice", "fund", frozenset({"risk"}))
    hits = search_documents(
        "What is the single-name exposure limit for Portfolio Alpha?",
        alice,
    )
    print([(hit["document_id"], round(hit["score"], 3)) for hit in hits])
