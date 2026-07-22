"""Persistent vector storage backed by ChromaDB."""

from dataclasses import dataclass

import chromadb

from agrimind.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, TOP_K
from agrimind.infrastructure.chunking.recursive_chunker import Chunk


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    text: str
    page_number: int
    document_id: str
    similarity: float


class ChromaStore:
    """Opens a persistent ChromaDB collection (cosine space) and upserts a
    document's chunks with provenance metadata under deterministic ids, so
    re-indexing the same document overwrites rather than duplicates.
    """

    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=_chunk_ids(document_id, chunks),
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {"document_id": document_id, "page_number": chunk.page_number}
                for chunk in chunks
            ],
        )

    def retrieve(self, query_embedding: list[float]) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            RetrievedChunk(
                text=document,
                page_number=metadata["page_number"],
                document_id=metadata["document_id"],
                similarity=1.0 - distance,
            )
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]


def _chunk_ids(document_id: str, chunks: list[Chunk]) -> list[str]:
    """Deterministic ids '<document_id>_p<page>_c<index>' where index counts
    chunks within their page, making re-indexing idempotent."""
    ids: list[str] = []
    per_page_count: dict[int, int] = {}
    for chunk in chunks:
        index = per_page_count.get(chunk.page_number, 0)
        per_page_count[chunk.page_number] = index + 1
        ids.append(f"{document_id}_p{chunk.page_number}_c{index}")
    return ids
