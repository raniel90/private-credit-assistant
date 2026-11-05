"""Read first-party files, prepare chunks, and sync them directly to Qdrant.

The manifest is the small lab's source of document identity and access metadata.
Qdrant is the derived search index. No SQLite intermediary is used.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from core import COLLECTION, EMBEDDING_MODEL, VECTOR_SIZE, FastEmbedder


ROOT = Path(__file__).parent
PIPELINE_VERSION = "text-v3-linewrap-600chars-100overlap"
QDRANT_URL = os.getenv("RAG_LAB_QDRANT_URL", "http://127.0.0.1:6333")


def read_text(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def prepare_chunks(text: str) -> list[str]:
    # Normalize whitespace before splitting; the original file remains the source.
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )
    return [part.strip() for part in splitter.split_text(normalized) if part.strip()]


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    existing = client.get_collection(COLLECTION).payload_schema
    for field in ("document_id", "tenant_id", "allowed_groups"):
        if field not in existing:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


def existing_points(client: QdrantClient, document_id: str) -> list:
    records = []
    offset = None
    doc_filter = models.Filter(
        must=[models.FieldCondition(
            key="document_id", match=models.MatchValue(value=document_id))]
    )
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=doc_filter,
            offset=offset,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            return records


def sync_document(client: QdrantClient, embedder: FastEmbedder, item: dict) -> str:
    path = (ROOT / item["source_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    chunks = prepare_chunks(read_text(path))
    if not chunks:
        raise ValueError(f"No readable content in {path}")

    document_id = item["document_id"]
    groups = sorted(set(item["allowed_groups"]))
    old = existing_points(client, document_id)
    expected_ids = {
        str(uuid5(NAMESPACE_URL, f"rag-lab/{document_id}/{index}"))
        for index in range(len(chunks))
    }
    metadata = {
        "document_id": document_id,
        "title": item["title"],
        "source_path": item["source_path"],
        "tenant_id": item["tenant_id"],
        "allowed_groups": groups,
        "sha256": checksum,
        "pipeline_version": PIPELINE_VERSION,
        "embedding_model": EMBEDDING_MODEL,
    }
    if {str(point.id) for point in old} == expected_ids and all(
        all(point.payload.get(key) == value for key, value in metadata.items())
        for point in old
    ):
        return f"{document_id}: unchanged ({len(chunks)} chunks)"

    vectors = embedder.embed(chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count")
    points = [
        models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"rag-lab/{document_id}/{index}")),
            vector=vector,
            payload={**metadata, "chunk_index": index, "text": chunk},
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    stale_ids = [point.id for point in old if str(
        point.id) not in expected_ids]
    if stale_ids:
        client.delete(collection_name=COLLECTION,
                      points_selector=stale_ids, wait=True)
    return f"{document_id}: indexed ({len(chunks)} chunks, removed {len(stale_ids)} stale)"


def main() -> None:
    manifest = json.loads(
        (ROOT / "data" / "documents.json").read_text(encoding="utf-8"))
    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client)
    embedder = FastEmbedder()
    for item in manifest:
        print(sync_document(client, embedder, item))
    print("Qdrant points:", client.count(
        collection_name=COLLECTION, exact=True).count)


if __name__ == "__main__":
    main()
