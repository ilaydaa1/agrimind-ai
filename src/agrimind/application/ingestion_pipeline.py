"""Application-layer ingestion pipeline: index a PDF end to end."""

from pathlib import Path

from agrimind.infrastructure.chunking.recursive_chunker import chunk_pages
from agrimind.infrastructure.embedding.e5_embedder import E5Embedder
from agrimind.infrastructure.extraction.pymupdf_extractor import extract_pages
from agrimind.infrastructure.vectorstore.chroma_store import ChromaStore


class IngestionPipeline:
    """Orchestrates extraction, chunking, embedding, and storage for one PDF.
    Owns the sequence only; every operation is delegated to an existing adapter.
    """

    def __init__(self, embedder: E5Embedder, store: ChromaStore) -> None:
        self._embedder = embedder
        self._store = store

    def index(self, pdf_path: Path) -> None:
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, self._embedder.count_tokens)
        embeddings = self._embedder.embed_chunks(chunks)
        self._store.add_document(pdf_path.stem, chunks, embeddings)
